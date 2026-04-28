"""
Pipeline 2 - Cross-lingual Knowledge Units
Tac dung:
  - Doc input tu transcript_v3.jsonl
  - Trich xuat keywords va topic
  - Tao Vietnamese keywords hints
  - Tao embedding_text voi Topic + Keywords + VI hints
  - Ghi ra transcript_v4.jsonl

Su dung:
  - config1.py: Cau hinh pipeline
  - utils1.py: Ham tien tro giup

Output: transcript_v4.jsonl voi embedding_text:
  Course: {course}
  Lecture: {title}

  Topic: {topic}

  Keywords: {en_keywords}
  Keywords (Vietnamese): {vi_keywords}

  Content:
  {chunk_text}
"""

from config1 import build_pipeline2_config
from utils1 import (
    iter_jsonl,
    write_jsonl,
    extract_keywords,
    get_vi_keywords,
    detect_topic,
    token_count,
)


def build_crosslingual_embedding(record: dict, config) -> str:
    """
    Tao embedding_text voi Cross-lingual format

    Format:
      Course: {course}
      Lecture: {title}

      Topic: {topic}

      Keywords: {en_keywords}
      Keywords (Vietnamese): {vi_keywords}

      Content:
      {chunk_text}

    Args:
      record: Dict chua thong tin chunk
      config: Cau hinh tu config1

    Returns:
      embedding_text: Chuoi da dinh dang cho embedding
    """
    # Lay thong tin
    chunk_text = record.get("chunk_text", "")
    course = record.get("course", "Unknown")
    title = record.get("title", "Unknown")

    # Trich xuat keywords tieng Anh
    keywords = extract_keywords(
        chunk_text,
        max_keywords=config.max_keywords,
        min_length=config.min_keyword_length,
    )

    # Dich sang tieng Viet
    vi_keywords = get_vi_keywords(keywords)

    # Phat hien topic chinh
    topic = detect_topic(chunk_text)

    # Tao embedding_text voi cac thanh phan
    lines = [
        f"Course: {course}",
        f"Lecture: {title}",
        "",
        f"Topic: {topic}",
        "",
        f"Keywords: {', '.join(keywords)}",
    ]

    # Them Vietnamese hints (neu duoc bat)
    if config.enable_vi_hints and vi_keywords:
        lines.append(f"Keywords (Vietnamese): {', '.join(vi_keywords)}")

    lines.append("")
    lines.append("Content:")
    lines.append(chunk_text)

    return "\n".join(lines)


def run_pipeline2():
    """
    Chay Pipeline 2

    Qua trinh:
      1. Doc cau hinh tu config1
      2. Doc tung record tu transcript_v3.jsonl
      3. Tao embedding_text moi (Topic + Keywords + VI hints)
      4. Ghi ra transcript_v4.jsonl
    """
    # Buoc 1: Lay cau hinh
    config = build_pipeline2_config()

    print("=" * 50)
    print("CROSS-LINGUAL KNOWLEDGE UNIT PIPELINE")
    print("=" * 50)
    print(f"Input: {config.input_path}")
    print(f"Output: {config.output_path}")
    print(f"VI hints: {config.enable_vi_hints}")
    print()

    all_records = []

    # Buoc 2-3: Xu ly tung record
    for record in iter_jsonl(config.input_path):
        # Tao embedding_text moi
        embedding_text = build_crosslingual_embedding(record, config)

        # Cap nhat record
        record["embedding_text"] = embedding_text
        record["chunk_type"] = "crosslingual_knowledge"

        all_records.append(record)

    # Buoc 4: Ghi output
    count = write_jsonl(all_records, config.output_path)

    print(f"Records: {count}")
    print(f"Output: {config.output_path}")


# CHAY PIPELINE
# python pipeline2.py
if __name__ == "__main__":
    run_pipeline2()
