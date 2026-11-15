import logging
import os
import time
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("huntsense-worker")

import processor
import scanner

def main():
    if not processor.supabase:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is required.")
    db = processor.supabase
    log.info("HuntSense combined worker started")

    while True:
        try:
            queued = (
                db.table("video_processing_jobs")
                .select("*")
                .eq("status", "queued")
                .order("created_at")
                .limit(1)
                .execute()
                .data
            )
            if queued:
                processor.process_job(queued[0])
                continue

            settings = (
                db.table("system_settings")
                .select("*")
                .eq("id", 1)
                .single()
                .execute()
                .data
            )
            if settings and settings.get("scanner_enabled") and scanner.model_available():
                pending = (
                    db.table("match_analyses")
                    .select("*")
                    .eq("status", "pending")
                    .order("created_at")
                    .limit(1)
                    .execute()
                    .data
                )
                if pending:
                    match = pending[0]
                    try:
                        scanner.analyze_match(match, False)
                    except Exception:
                        log.exception("Analysis failed for match %s", match.get("id"))
                        db.table("match_analyses").update({"status": "failed"}).eq("id", match["id"]).execute()
                    continue

            time.sleep(5)
        except Exception:
            log.exception("Worker loop error; retrying in 10 seconds")
            time.sleep(10)

if __name__ == "__main__":
    main()
