from langchain_core.documents import Document
import json

docs = []

with open('/Users/carwyn/Downloads/transcript_v3.jsonl') as f:
    for line in f:
        chunk = json.loads(line)

        doc = Document(
            page_content=chunk["chunk_text"],
            metadata={
                "video_id": chunk["video_id"],
                "title": chunk["title"],
                "course": chunk["course"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "duration": chunk["duration"],
                "source": chunk["source"],
                "chunk_type": chunk["chunk_type"],
                "token_count": chunk["token_count"],
            }
        )

        docs.append(doc)
print("Số lượng document:", len(docs))

print("\nSample 1 document:")
print(docs[0].page_content[:200])  # xem 200 ký tự đầu
print(docs[0].metadata)