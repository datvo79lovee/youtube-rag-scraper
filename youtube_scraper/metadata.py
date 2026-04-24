import requests
from config import HTTP_TIMEOUT
from youtube_scraper.utils import extract_playlist_id


def get_playlist_videos(api_key, playlist_url):
    playlist_id = extract_playlist_id(playlist_url)
    videos = []
    next_page_token = None

    while True:
        url = "https://www.googleapis.com/youtube/v3/playlistItems"
        params = {
            "part": "snippet",
            "playlistId": playlist_id,
            "maxResults": 50,
            "key": api_key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        res = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        res.raise_for_status()
        data = res.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        for item in data.get("items", []):
            snippet = item["snippet"]
            if snippet["resourceId"]["kind"] != "youtube#video":
                continue

            videos.append({
                "video_id": snippet["resourceId"]["videoId"],
                "title": snippet["title"],
                "playlist_id": playlist_id,
                "published_at": snippet.get("publishedAt"),
            })

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return list({v["video_id"]: v for v in videos}.values())
