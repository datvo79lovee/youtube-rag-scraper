import logging
import os
import random
import time

from config import (
    BLOCK_COOLDOWN_SECONDS,
    BLOCK_STREAK_THRESHOLD,
    FAILED_CACHE_FILE,
    LONG_BREAK_EVERY_N_SUCCESS,
    LONG_BREAK_SECONDS,
    MAX_RETRIES,
    MIN_SEGMENTS,
    PLAYLISTS,
    RETRY_DELAY,
    SLEEP_BETWEEN_CALLS_MAX,
    SLEEP_BETWEEN_CALLS_MIN,
    SUCCESS_CACHE_FILE,
    TARGET_MAX,
    YOUTUBE_API_KEY,
)
from youtube_scraper.knowledge_base import append_jsonl
from youtube_scraper.metadata import get_playlist_videos
from youtube_scraper.transcripts import fetch_transcript
from youtube_scraper.utils import load_processed_ids, save_processed_id


logging.basicConfig(level=logging.INFO)


def run():
    os.makedirs("assets", exist_ok=True)

    success_ids = load_processed_ids(SUCCESS_CACHE_FILE)
    failed_ids = load_processed_ids(FAILED_CACHE_FILE)

    sleep_min = max(SLEEP_BETWEEN_CALLS_MIN, 1.0)
    sleep_max = max(SLEEP_BETWEEN_CALLS_MAX, sleep_min)

    total = 0
    success = 0
    skipped = 0
    failed = 0
    consecutive_blocked = 0

    for i, pl in enumerate(PLAYLISTS):
        name = pl["name"]
        url = pl["url"]

        logging.info(f"PLAYLIST {i + 1}/{len(PLAYLISTS)}: {name}")
        try:
            videos = get_playlist_videos(YOUTUBE_API_KEY, url)
        except Exception as e:
            logging.error(f"FAILED TO LOAD PLAYLIST {name}: {e}")
            time.sleep(max(RETRY_DELAY, 10))
            continue

        for v in videos:
            vid = v["video_id"]

            if vid in success_ids:
                continue

            logging.info(f"[{total + skipped + failed}] Processing {vid}")

            try:
                result = fetch_transcript(
                    vid,
                    min_segments=MIN_SEGMENTS,
                    retries=MAX_RETRIES,
                    delay=RETRY_DELAY,
                )

                status = result.get("status", "error")

                if status == "success":
                    record = {
                        "video_id": vid,
                        "title": v["title"],
                        "playlist_id": v["playlist_id"],
                        "course": name,
                        "source": "stanford_youtube",
                        "language": result["language"],
                        "is_auto_generated": result["is_auto_generated"],
                        "published_at": v["published_at"],
                        "transcript": result["transcript"],
                    }

                    append_jsonl(record)
                    save_processed_id(SUCCESS_CACHE_FILE, vid)
                    success_ids.add(vid)

                    success += 1
                    total += 1
                    consecutive_blocked = 0

                    if total % LONG_BREAK_EVERY_N_SUCCESS == 0:
                        logging.info("Taking a longer break...")
                        time.sleep(LONG_BREAK_SECONDS)

                    logging.info(f"SUCCESS {vid} | total={total}")

                    if total >= TARGET_MAX:
                        logging.info("Reached TARGET_MAX")
                        return

                    time.sleep(random.uniform(sleep_min, sleep_max))
                    continue

                if status in {
                    "no_en",
                    "too_short",
                    "no_transcript",
                    "transcripts_disabled",
                }:
                    if vid not in failed_ids:
                        save_processed_id(FAILED_CACHE_FILE, vid)
                        failed_ids.add(vid)

                    skipped += 1
                    consecutive_blocked = 0
                    logging.info(f"SKIP {vid} | reason={status}")
                    time.sleep(random.uniform(sleep_min, sleep_max))
                    continue

                failed += 1

                if status == "blocked":
                    consecutive_blocked += 1
                    logging.warning(
                        f"BLOCKED {vid} | consecutive_blocked={consecutive_blocked} | "
                        f"error={result.get('error', 'unknown')}"
                    )
                else:
                    consecutive_blocked = 0
                    logging.error(
                        f"ERROR {vid} | status={status} | "
                        f"error={result.get('error', 'unknown')}"
                    )

                if consecutive_blocked >= BLOCK_STREAK_THRESHOLD:
                    logging.warning(
                        "Detected too many consecutive blocked transcript requests. "
                        f"Cooling down for {BLOCK_COOLDOWN_SECONDS}s and stopping run."
                    )
                    time.sleep(BLOCK_COOLDOWN_SECONDS)
                    return

                time.sleep(random.uniform(sleep_min, sleep_max))

            except Exception as e:
                failed += 1
                consecutive_blocked = 0
                logging.error(f"FATAL {vid}: {e}")
                time.sleep(max(RETRY_DELAY, sleep_min))
                continue

    logging.info(f"DONE | success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    run()
