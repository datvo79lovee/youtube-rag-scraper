import os

def load_processed_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_processed_id(path, vid):
    with open(path, "a", encoding="utf-8") as f:
        f.write(vid.strip() + "\n")



def extract_playlist_id(url: str):
    return url.split("list=")[-1].split("&")[0]