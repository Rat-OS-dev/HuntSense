import os, re, time, json, tempfile, subprocess, logging
from pathlib import Path
import cv2
import pytesseract
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
            supabase.storage.from_(BUCKET).upload(storage,out.read_bytes(),{"content-type":"video/mp4","upsert":"false"})
            row=supabase.table("match_analyses").insert({
                "user_id":uid,"video_path":storage,
                "video_title":f'{Path(job["source_title"]).stem} - Match {idx}',
                "status":"pending","timestamps":[]
            }).execute().data[0]
            created.append(row["id"])
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
