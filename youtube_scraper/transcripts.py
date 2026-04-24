import time
from youtube_transcript_api import YouTubeTranscriptApi

BLOCK_HINTS = ("429", "too many requests", "request blocked", "unusual traffic", "ip blocked")


def classify_transcript_error(exc):
    msg = str(exc).lower()
    if any(hint in msg for hint in BLOCK_HINTS):
        return "blocked"
    if "transcript is disabled" in msg:
        return "transcripts_disabled"
    if "no transcripts were found" in msg or "no transcript" in msg:
        return "no_transcript"
    return "error"


def fetch_transcript(video_id, min_segments=50, retries=3, delay=5):
    ytt = YouTubeTranscriptApi()
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            t_list = ytt.list(video_id)
            en_tracks = [t for t in t_list if t.language_code.startswith("en")]

            if not en_tracks:
                return {"status": "no_en"}

            manual = [t for t in en_tracks if not t.is_generated]
            transcript = manual[0] if manual else en_tracks[0]
            is_auto = not bool(manual)

            raw_segments = transcript.fetch()
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in raw_segments
            ]

            if not segments or len(segments) < min_segments:
                return {"status": "too_short", "segment_count": len(segments)}

            return {
                "status": "success",
                "language": transcript.language_code,
                "is_auto_generated": is_auto,
                "transcript": segments,
            }

        except Exception as e:
            last_error = str(e)
            status = classify_transcript_error(e)
            if status in {"blocked", "transcripts_disabled", "no_transcript"}:
                return {"status": status, "error": last_error}
            if attempt < retries:
                time.sleep(delay * attempt)

    return {"status": "error", "error": last_error or "unknown"}
