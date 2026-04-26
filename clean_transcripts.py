import json
import re
import os

FILLER_WORDS = {
    "ok",
    "yeah",
    "uh",
    "um",
    "oh",
    "ah",
    "hey",
    "alright",
    "so",
    "like",
    "actually",
    "literally",
    "basically",
    "totally",
    "i mean",
    "you know",
    "i guess",
    "i suppose",
}

MARKERS = ["[INAUDIBLE]", "[APPLAUSE]", "[LAUGHTER]", "[MUSIC]", "[SILENCE]"]


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")

    for marker in MARKERS:
        text = text.replace(marker, "")

    text = re.sub(r"\[.*?\]", "", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip(" -.,")

    return text


def remove_fillers(text: str) -> str:
    words = text.split()
    filtered = []
    skip_next = 0

    for i, word in enumerate(words):
        if skip_next > 0:
            skip_next -= 1
            continue

        lower = word.lower().rstrip(".,!?")

        if lower in FILLER_WORDS:
            if lower in ["i guess", "i mean", "you know"]:
                skip_next = 1
            continue

        filtered.append(word)

    return " ".join(filtered)


def timestamp_chunk(
    transcript: list, max_duration: int = 30, min_words: int = 10
) -> list:
    chunks = []

    if not transcript:
        return chunks

    current_text = []
    chunk_start = None
    chunk_words = 0

    for seg in transcript:
        text = clean_text(seg["text"])
        if not text:
            continue

        words = text.split()

        if chunk_start is None:
            chunk_start = seg["start"]

        chunk_words += len(words)
        current_text.append(text)

        end_time = seg["start"] + seg["duration"]
        duration = end_time - chunk_start

        if duration >= max_duration or chunk_words >= min_words * 2:
            if current_text:
                full_text = " ".join(current_text)
                full_text = remove_fillers(full_text)
                full_text = clean_text(full_text)

                if len(full_text.split()) >= min_words:
                    chunks.append(
                        {
                            "start": round(chunk_start, 2),
                            "end": round(end_time, 2),
                            "text": full_text,
                        }
                    )

            current_text = []
            chunk_start = None
            chunk_words = 0

    if current_text:
        full_text = " ".join(current_text)
        full_text = remove_fillers(full_text)
        full_text = clean_text(full_text)

        if len(full_text.split()) >= min_words:
            chunks.append(
                {
                    "start": round(chunk_start, 2),
                    "end": round(
                        transcript[-1]["start"] + transcript[-1]["duration"], 2
                    ),
                    "text": full_text,
                }
            )

    return chunks


def extract_topic(title: str) -> str:
    parts = title.split("I ")
    if len(parts) > 1:
        topic = parts[1].strip()
        topic = re.sub(r"^[\w\s]+,\s*", "", topic)
        topic = re.sub(r"\s+\w+ \d{4}$", "", topic)
        return topic.strip()

    if "V2" in title:
        return title.split("V2")[1].strip()
    if "V3" in title:
        return title.split("V3")[1].strip()
    if "V4" in title:
        return title.split("V4")[1].strip()
    if "V5" in title:
        return title.split("V5")[1].strip()

    return title


def process_transcript(record: dict) -> dict:
    topic = extract_topic(record.get("title", ""))

    raw_segments = record.get("transcript", [])

    chunks = timestamp_chunk(raw_segments, max_duration=30, min_words=10)

    return {
        "video_id": record["video_id"],
        "title": record["title"],
        "topic": topic,
        "course": record.get("course", ""),
        "published_at": record.get("published_at", ""),
        "source": record.get("source", ""),
        "chunks": chunks,
    }


def clean_transcripts(input_path: str, output_path: str):
    processed = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                processed_record = process_transcript(record)
                processed.append(processed_record)
            except json.JSONDecodeError:
                print(f"SKIP: invalid JSON")
                continue

    with open(output_path, "w", encoding="utf-8") as f:
        for record in processed:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done: {len(processed)} videos processed")
    print(f"Output: {output_path}")

    stats = {
        "total_videos": len(processed),
        "total_chunks": sum(len(r["chunks"]) for r in processed),
        "avg_chunks_per_video": sum(len(r["chunks"]) for r in processed)
        / len(processed)
        if processed
        else 0,
    }
    print(f"Stats: {stats}")


if __name__ == "__main__":
    input_file = r"D:\youtube-rag-scraper\assets\transcripts.jsonl"
    output_file = r"D:\youtube-rag-scraper\assets\transcripts_clean.jsonl"

    clean_transcripts(input_file, output_file)
