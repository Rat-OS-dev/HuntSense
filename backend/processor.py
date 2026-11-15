import os, re, time, json, tempfile, subprocess, logging
from pathlib import Path
import cv2
import pytesseract
import requests
import base64
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
log=logging.getLogger("huntsense-processor")
URL=os.getenv("SUPABASE_URL","https://rtyzudbbwoulrysvypql.supabase.co")
KEY=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
BUCKET="huntsense-videos"
MAX_SECONDS=10*60*60
supabase=create_client(URL, KEY) if KEY else None

def run(cmd):
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:])
    return result.stdout

def download_source(path,target):
    signed=supabase.storage.from_(BUCKET).create_signed_url(path,3600)
    url=signed.get("signedURL") or signed.get("signedUrl")
    if not url: raise RuntimeError("Could not create source download URL")
    with requests.get(url,stream=True,timeout=(30,300)) as response:
        response.raise_for_status()
        with open(target,"wb") as out:
            for chunk in response.iter_content(8*1024*1024):
                if chunk: out.write(chunk)

def tus_upload(local_path, object_path, content_type="video/mp4"):
    endpoint=URL.rstrip("/")+"/storage/v1/upload/resumable"
    headers={"apikey":KEY,"Authorization":"Bearer "+KEY,"Tus-Resumable":"1.0.0",
             "Upload-Length":str(os.path.getsize(local_path)),
             "Upload-Metadata":",".join(f"{k} {base64.b64encode(v.encode()).decode()}" for k,v in {
                 "bucketName":BUCKET,"objectName":object_path,"contentType":content_type,"cacheControl":"3600"
             }.items())}
    response=requests.post(endpoint,headers=headers,timeout=60)
    if response.status_code not in (201,202): raise RuntimeError("TUS create failed: "+response.text[:1000])
    location=response.headers.get("Location")
    if not location: raise RuntimeError("TUS server did not return an upload location")
    if location.startswith("/"): location=URL.rstrip("/")+location
    offset=0
    with open(local_path,"rb") as source:
        while True:
            chunk=source.read(6*1024*1024)
            if not chunk: break
            patch_headers={"apikey":KEY,"Authorization":"Bearer "+KEY,"Tus-Resumable":"1.0.0",
                           "Upload-Offset":str(offset),"Content-Type":"application/offset+octet-stream"}
            patch=requests.patch(location,headers=patch_headers,data=chunk,timeout=300)
            if patch.status_code not in (204,): raise RuntimeError("TUS chunk upload failed: "+patch.text[:1000])
            offset=int(patch.headers.get("Upload-Offset",offset+len(chunk)))
    return True

def probe(path):
    raw=run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path])
    return float(raw.strip())

def text_at(frame):
    h,w=frame.shape[:2]
    frame=cv2.resize(frame,(min(1280,w),int(h*min(1280,w)/w)))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)[1]
    return re.sub(r"[^A-Z0-9 ]"," ",pytesseract.image_to_string(gray).upper())

def find_rounds(path):
    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration=total/fps if total else 0
    starts=[]
    ends=[]
    frame_no=0
    last_sample=-10
    while True:
        ok,frame=cap.read()
        if not ok: break
        sec=frame_no/fps
        frame_no+=1
        if sec-last_sample<2: continue
        last_sample=sec
        text=text_at(frame)
        compact=" ".join(text.split())
        if "RISE UP" in compact and ("DAMNED" in compact or "SOULS" in compact):
            if not starts or sec-starts[-1]>180:
                starts.append(sec)
        # Lobby detection deliberately requires a strong cluster of lobby terms
        # and only applies after a round has started.
        lobby_terms=sum(term in compact for term in ("RETURN TO LOBBY","BACK TO LOBBY","MATCH SUMMARY","MISSION SUMMARY","PLAY AGAIN"))
        if lobby_terms>=1 and starts and sec>starts[-1]+120:
            if not ends or sec-ends[-1]>120:
                ends.append(sec)
    cap.release()
    rounds=[]
    for i,start in enumerate(starts):
        next_start=starts[i+1] if i+1<len(starts) else duration
        possible=[e for e in ends if start+120<e<next_start]
        end=possible[0] if possible else next_start
        # Small handles preserve the round intro and end transition.
        a=max(0,start-4)
        b=min(duration,end+2)
        if b-a>=120:
            rounds.append((a,b))
    return duration,rounds

def encode_segment(source,start,end,target):
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str(start),"-i",source,"-t",str(end-start),
         "-vf","scale='min(1920,iw)':-2:force_original_aspect_ratio=decrease",
         "-c:v","libx265","-preset","medium","-crf","27","-tag:v","hvc1",
         "-c:a","aac","-b:a","96k","-movflags","+faststart",target])

def detect_fights(path, model_path):
    from ultralytics import YOLO
    detector=YOLO(model_path)
    cap=cv2.VideoCapture(path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    duration=probe(path)
    events=[]
    sec=0.0
    while sec<duration:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ok,frame=cap.read()
        if not ok: break
        result=detector(frame,verbose=False)[0]
        names=result.names
        found=False
        for box in result.boxes:
            label=str(names[int(box.cls[0])]).lower().replace(" ","_")
            if label in ("enemy_hunter","enemy","hunter_enemy") and float(box.conf[0])>=0.70:
                found=True
                break
        if found:
            if events and sec<=events[-1][1]+10:
                events[-1]=(events[-1][0],min(duration,sec+50))
            else:
                events.append((max(0,sec-20),min(duration,sec+50)))
        sec+=1.0
    cap.release()
    return events

def save_fight_clips(match_id,user_id,match_path,local_path,model_path):
    events=detect_fights(local_path,model_path)
    saved=[]
    temp=Path(local_path).parent
    for index,(start,end) in enumerate(events,1):
        clip=temp/f"fight-{match_id}-{index}.mp4"
        run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str(start),"-i",local_path,
             "-t",str(max(1,end-start)),"-c:v","libx265","-preset","fast","-crf","28",
             "-c:a","aac","-b:a","96k","-movflags","+faststart",str(clip)])
        path=f"{user_id}/fight-clips/{match_id}/{index:03d}.mp4"
        tus_upload(str(clip),path)
        supabase.table("video_fight_clips").insert({
            "match_id":match_id,"user_id":user_id,"storage_path":path,
            "start_sec":start,"end_sec":end
        }).execute()
        saved.append({"start_sec":round(start,1),"end_sec":round(end,1)})
        clip.unlink(missing_ok=True)
    supabase.table("match_analyses").update({"timestamps":saved}).eq("id",match_id).execute()
    return len(saved)

def process_job(job):
    jid=job["id"]; uid=job["user_id"]; src=job["source_path"]
    work=Path(tempfile.mkdtemp(prefix="hs-"))
    original=work/"source"
    try:
        supabase.table("video_processing_jobs").update({"status":"processing","error_message":None}).eq("id",jid).execute()
        data=supabase.storage.from_(BUCKET).download(src)
        original.write_bytes(data)
        duration=probe(str(original))
        if duration>MAX_SECONDS+3:
            raise ValueError("Recording exceeds the 10-hour maximum.")
        supabase.table("video_processing_jobs").update({"duration_seconds":duration}).eq("id",jid).execute()
        duration,rounds=find_rounds(str(original))
        if not rounds:
            raise ValueError("No round-start screens were confidently detected. Original recording was kept for review.")
        created=[]
        for idx,(start,end) in enumerate(rounds,1):
            out=work/f"match-{idx:02d}.mp4"
            encode_segment(str(original),start,end,str(out))
            storage=f"{uid}/matches/{jid}/match-{idx:02d}.mp4"
            tus_upload(str(out),storage)
            row=supabase.table("match_analyses").insert({
                "user_id":uid,"video_path":storage,
                "video_title":f'{Path(job["source_title"]).stem} - Match {idx}',
                "status":"pending","timestamps":[]
            }).execute().data[0]
            created.append(row["id"])\n            model_path=os.getenv("YOLO_MODEL_PATH")\n            if model_path and Path(model_path).exists():\n                count=save_fight_clips(row["id"],uid,storage,str(out),model_path)\n                log.info("Saved %s fight clips for match %s",count,row["id"])
        # Never remove the source until every match has encoded, uploaded, and a DB row exists.
        supabase.storage.from_(BUCKET).remove([src])
        supabase.table("video_processing_jobs").update({
            "status":"completed","error_message":None
        }).eq("id",jid).execute()
        log.info("Job %s produced %s match videos",jid,len(created))
    except Exception as exc:
        log.exception("Job %s failed",jid)
        supabase.table("video_processing_jobs").update({
            "status":"failed","error_message":str(exc)[:1500]
        }).eq("id",jid).execute()
    finally:
        for p in work.glob("*"):
            try: p.unlink()
            except OSError: pass
        try: work.rmdir()
        except OSError: pass

def main():
    if not supabase:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is required.")
    log.info("HuntSense long-recording processor started")
    while True:
        result=supabase.table("video_processing_jobs").select("*").eq("status","queued").order("created_at").limit(1).execute()
        if result.data:
            process_job(result.data[0])
        else:
            time.sleep(8)

if __name__=="__main__":
    main()
