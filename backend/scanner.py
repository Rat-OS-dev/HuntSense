import os, time, subprocess, random, json, base64, logging, requests
from pathlib import Path
import cv2
import numpy as np
import easyocr
from dotenv import load_dotenv
from supabase import create_client
from ultralytics import YOLO
from google import genai

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("huntsense-scanner")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://rtyzudbbwoulrysvypql.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BUCKET = "huntsense-videos"

if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is required.")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
YOLO_MODEL_STORAGE_PATH = os.getenv("YOLO_MODEL_STORAGE_PATH")
model = None
ocr_reader = None

def model_available():
    return bool((MODEL_PATH and Path(MODEL_PATH).is_file()) or YOLO_MODEL_STORAGE_PATH)

def load_model():
    global model, MODEL_PATH
    if model is not None:
        return model
    path = MODEL_PATH
    if not path or not Path(path).is_file():
        if not YOLO_MODEL_STORAGE_PATH:
            raise RuntimeError("A trained HuntSense YOLO .pt model is required.")
        path = "/tmp/huntsense-detector.pt"
        if not Path(path).is_file():
            Path(path).write_bytes(supabase.storage.from_(BUCKET).download(YOLO_MODEL_STORAGE_PATH))
        MODEL_PATH = path
    model = YOLO(path)
    return model

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        ocr_reader = easyocr.Reader(["en"], gpu=os.getenv("OCR_GPU", "false").lower() == "true")
    return ocr_reader

def tus_upload(local_path, object_path, content_type="video/mp4"):
    storage_url = SUPABASE_URL.replace(".supabase.co", ".storage.supabase.co")
    endpoint = storage_url.rstrip("/") + "/storage/v1/upload/resumable"
    metadata = {"bucketName": BUCKET, "objectName": object_path, "contentType": content_type, "cacheControl": "3600"}
    headers = {
        "apikey": SUPABASE_SERVICE_KEY, "Authorization": "Bearer " + SUPABASE_SERVICE_KEY,
        "Tus-Resumable": "1.0.0", "Upload-Length": str(os.path.getsize(local_path)), "x-upsert": "true",
        "Upload-Metadata": ",".join(f"{k} {base64.b64encode(v.encode()).decode()}" for k,v in metadata.items())
    }
    created = requests.post(endpoint, headers=headers, timeout=60)
    if created.status_code not in (201, 202):
        raise RuntimeError("Fight-clip upload could not start: " + created.text[:800])
    location = created.headers.get("Location")
    if not location:
        raise RuntimeError("Storage did not return a resumable upload URL.")
    if location.startswith("/"):
        location = storage_url.rstrip("/") + location
    offset = 0
    with open(local_path, "rb") as source:
        while True:
            chunk = source.read(6 * 1024 * 1024)
            if not chunk:
                break
            patch = requests.patch(location, headers={
                "apikey": SUPABASE_SERVICE_KEY, "Authorization": "Bearer " + SUPABASE_SERVICE_KEY,
                "Tus-Resumable": "1.0.0", "Upload-Offset": str(offset),
                "Content-Type": "application/offset+octet-stream"
            }, data=chunk, timeout=300)
            if patch.status_code != 204:
                raise RuntimeError("Fight-clip upload failed: " + patch.text[:800])
            offset = int(patch.headers.get("Upload-Offset", offset + len(chunk)))

def compress_video(raw_path, out_path):
    """Super-compress match video to 720p H.265 so it streams smoothly from anywhere."""
    cmd = [
        "ffmpeg", "-y", "-i", raw_path,
        "-vf", "scale=-2:720", "-c:v", "libx265", "-crf", "28", "-preset", "fast",
        "-c:a", "aac", "-b:a", "96k", out_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("Video compression failed: " + result.stderr[-1500:])

def cut_clip(video_path, start_sec, end_sec, clip_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(max(0, start_sec)), "-i", video_path,
        "-t", str(max(1, end_sec - start_sec)), "-c", "copy", clip_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def detect_mouse_chop_ms(frames_gray, enemy_center_x, frame_w, fps):
    """Measures ms from enemy appearing to noticeable fast camera chop toward the enemy."""
    if len(frames_gray) < 4:
        return None
    prev = frames_gray[0]
    for idx in range(1, len(frames_gray)):
        curr = frames_gray[idx]
        # Optical flow in center box to measure horizontal mouse pan speed
        h, w = curr.shape
        roi_prev = prev[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        roi_curr = curr[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        flow = cv2.calcOpticalFlowFarneback(roi_prev, roi_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mean_dx = float(np.mean(flow[..., 0]))
        # Noticeable fast chop (> 6 pixels/frame shift)
        if abs(mean_dx) > 6.0:
            return int((idx / fps) * 1000)
        prev = curr
    return None

def analyze_match(match_row, track_zombies):
    match_id = match_row["id"]
    detector = load_model()
    reader = get_ocr_reader()
    supabase.table("match_analyses").update({"status": "scanning"}).eq("id", match_id).execute()

    raw_video = f"/tmp/{match_id}_raw.mp4"
    compressed_video = f"/tmp/{match_id}_web.mp4"
    data = supabase.storage.from_(BUCKET).download(match_row["video_path"])
    with open(raw_video, "wb") as f:
        f.write(data)

    cap = cv2.VideoCapture(raw_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fights = []
    active_fight = None
    last_enemy_sec = -999
    fight_gun_counts = {}
    shots_hit = 0
    recent_gray = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        sec = frame_idx / fps
        frame_idx += 1

        # Sample every 3rd frame for fast scanning while keeping millisecond accuracy
        if frame_idx % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        recent_gray.append(gray)
        if len(recent_gray) > int(fps):
            recent_gray.pop(0)

        results = detector(frame, verbose=False)[0]
        enemy_found = False
        enemy_x = frame_w // 2

        for box in results.boxes:
            cls_name = detector.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # Confidence Active Learning Queue (70-89% + 5% random low-confidence)
            if (0.70 <= conf < 0.90) or (0.25 <= conf < 0.70 and random.random() < 0.05):
                try:
                    snap_name = f"{match_row['user_id']}/review/{match_id}_{frame_idx}.jpg"
                    _, buf = cv2.imencode(".jpg", frame)
                    supabase.storage.from_(BUCKET).upload(snap_name, buf.tobytes(), {"content-type": "image/jpeg"})
                    supabase.table("yolo_review_queue").insert({
                        "image_url": snap_name, "predicted_label": cls_name,
                        "confidence": round(conf, 2), "bbox": box.xyxy[0].tolist()
                    }).execute()
                except Exception as queue_error:
                    log.debug("Optional review queue is not configured: %s", queue_error)

            if cls_name == "enemy_hunter" and conf >= 0.70:
                enemy_found = True
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                enemy_x = (x1 + x2) / 2
            elif cls_name == "hit_marker" and conf >= 0.75:
                shots_hit += 1

        # PvP Rolling Clip Timer: Lock -20s start, restart 50s countdown on every sighting
        if enemy_found:
            if active_fight is None:
                reaction_ms = detect_mouse_chop_ms(recent_gray[-15:], enemy_x, frame_w, fps / 3)
                active_fight = {
                    "start_sec": round(max(0, sec - 20.0), 1),
                    "end_sec": round(sec + 50.0, 1),
                    "reaction_ms": reaction_ms,
                    "notes": "Enemy Hunter spotted"
                }
            else:
                # Restart 50s countdown while keeping original start_sec locked
                active_fight["end_sec"] = round(sec + 50.0, 1)
            last_enemy_sec = sec

            # Check most used gun ONLY during active fight clips
            hud_crop = frame[int(frame_h * 0.85):frame_h, int(frame_w * 0.75):frame_w]
            texts = reader.readtext(hud_crop, detail=0)
            for t in texts:
                if len(t) > 3:
                    fight_gun_counts[t] = fight_gun_counts.get(t, 0) + 1

        elif active_fight is not None and sec > active_fight["end_sec"]:
            fights.append(active_fight)
            active_fight = None

    if active_fight is not None:
        fights.append(active_fight)
    cap.release()

    # Compress full match for web streaming & upload back to Supabase
    compress_video(raw_video, compressed_video)
    tus_upload(compressed_video, match_row["video_path"])

    # Save each detected fight clip to the private video bucket and its table.
    for idx, fight in enumerate(fights, 1):
        clip_file = f"/tmp/{match_id}_fight_{idx}.mp4"
        cut_clip(compressed_video, fight["start_sec"], fight["end_sec"], clip_file)
        fight["notes"] = "AI notes are not configured."
        if ai_client:
            try:
                uploaded = ai_client.files.upload(file=clip_file)
                resp = ai_client.models.generate_content(
                    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                    contents=[uploaded, "Summarize only observable events in this gameplay clip in one concise sentence."]
                )
                fight["notes"] = (resp.text or "").strip() or fight["notes"]
                try:
                    ai_client.files.delete(name=uploaded.name)
                except Exception:
                    pass
            except Exception as ai_error:
                log.warning("Optional AI clip summary failed: %s", ai_error)
        clip_path = f"{match_row['user_id']}/fight-clips/{match_id}/{idx:03d}.mp4"
        tus_upload(clip_file, clip_path)
        supabase.table("video_fight_clips").insert({
            "match_id": match_id, "user_id": match_row["user_id"],
            "storage_path": clip_path, "start_sec": fight["start_sec"],
            "end_sec": fight["end_sec"], "analysis_notes": fight["notes"]
        }).execute()
        os.remove(clip_file)

    best_gun = max(fight_gun_counts, key=fight_gun_counts.get) if fight_gun_counts else "Not identified yet"
    valid_rx = [f["reaction_ms"] for f in fights if f["reaction_ms"]]
    avg_rx = int(sum(valid_rx) / len(valid_rx)) if valid_rx else None
    est_shots_fired = max(shots_hit, len(fights) * 3)
    hit_pct = round((shots_hit / est_shots_fired) * 100, 1) if est_shots_fired else 0

    supabase.table("match_analyses").update({
        "status": "ready",
        "timestamps": fights,
        "shots_fired": est_shots_fired,
        "shots_hit": shots_hit,
        "shots_hit_pct": hit_pct,
        "most_used_fight_gun": best_gun,
        "avg_reaction_ms": avg_rx
    }).eq("id", match_id).execute()

    os.remove(raw_video)
    os.remove(compressed_video)

def main_loop():
    print("HuntSense Linux Worker running. Checking Admin Master Scanner toggle...")
    while True:
        settings = supabase.table("system_settings").select("*").eq("id", 1).single().execute().data
        if settings and settings.get("scanner_enabled") and model_available():
            pending = supabase.table("match_analyses").select("*").eq("status", "pending").limit(1).execute().data
            if pending:
                analyze_match(pending[0], False)
                continue
        time.sleep(5)

if __name__ == "__main__":
    main_loop()
