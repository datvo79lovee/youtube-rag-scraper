import json

from config import YOUTUBE_API_KEY
from youtube_scraper.metadata import get_playlist_videos
from youtube_scraper.transcripts import fetch_transcript


playlist_url = "https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599"

videos = get_playlist_videos(YOUTUBE_API_KEY, playlist_url)

for v in videos[:5]:
    vid = v["video_id"]
    print(f"\n=== VIDEO: {vid} ===")

    result = fetch_transcript(vid, min_segments=10)
    status = result.get("status", "error")

    print("STATUS:", status)

    if status != "success":
        print("DETAIL:", result.get("error", status))
        continue

    print("LANG:", result["language"])
    print("AUTO:", result["is_auto_generated"])

    print("\nSample segments:")
    for seg in result["transcript"][:3]:
        print(json.dumps(seg, indent=2, ensure_ascii=False))
