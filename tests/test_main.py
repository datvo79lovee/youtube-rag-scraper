import json

from config import MIN_SEGMENTS, PLAYLISTS, YOUTUBE_API_KEY
from youtube_scraper.metadata import get_playlist_videos
from youtube_scraper.transcripts import fetch_transcript


def test_crawl():
    playlist = PLAYLISTS[7]
    playlist_url = playlist["url"]
    playlist_name = playlist["name"]

    print(f"\n=== TEST PLAYLIST: {playlist_name} ===")

    videos = get_playlist_videos(YOUTUBE_API_KEY, playlist_url)

    if not videos:
        print("No videos returned from playlist API")
        return

    print(f"Total videos: {len(videos)}")

    results = []

    for vid in videos[:2]:
        video_id = vid["video_id"]
        title = vid["title"]

        print(f"\n=== VIDEO: {video_id} ===")
        print(f"Title: {title}")

        data = fetch_transcript(video_id, min_segments=MIN_SEGMENTS)
        status = data.get("status", "error")

        print(f"Status: {status}")

        if status != "success":
            print(f"Skip reason: {data.get('error', status)}")
            continue

        print(f"Segments: {len(data['transcript'])}")
        print(f"Lang: {data['language']} | Auto: {data['is_auto_generated']}")

        print("Sample:")
        for seg in data["transcript"][:3]:
            print(seg)

        results.append(
            {
                "video_id": video_id,
                "title": title,
                "playlist": playlist_name,
                "transcript_result": data,
            }
        )

    with open("test_output.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nDone test. Saved to test_output.jsonl")


if __name__ == "__main__":
    test_crawl()
