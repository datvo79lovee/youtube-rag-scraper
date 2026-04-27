from langchain_community.document_loaders import JSONLoader

def load_chunked_json(file_path: str):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".",          # mỗi dùng là 1 object
        content_key="chunk_text",        # field chứa nội dung
        metadata_func=metadata_func, # metadata_func để trích xuất metadata từ record 
        json_lines=True    
    )
        

    docs = loader.load()
    return docs

def metadata_func(record, metadata):
    metadata["chunk_id"] = record.get("chunk_id")
    metadata["start_time"] = record.get("start_time")
    metadata["end_time"] = record.get("end_time")
    return metadata

path = '/Users/carwyn/youtube-rag-scraper/embed/files/transcripts_enhanced.jsonl'
docs = load_chunked_json(path)
print (len(docs))

from pprint import pprint
for i in range (3):
    pprint (docs[i].page_content)
    pprint (docs[i].metadata)
