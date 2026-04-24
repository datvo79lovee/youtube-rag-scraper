from youtube_scraper.transcripts import fetch_transcript


video_id = "kCc8FmEb1nY"

result = fetch_transcript(video_id, min_segments=10)

print("STATUS:", result.get("status", "error"))
print("RESULT:", result)
