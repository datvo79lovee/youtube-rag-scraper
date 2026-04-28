from langchain_community.vectorstores import FAISS
from core.embeddings import get_embedding_model
import pickle

# Load FAISS index
embedder = get_embedding_model()
vs = FAISS.load_local(
    "index/faiss",
    embedder,
    allow_dangerous_deserialization=True
)

# Semantic search
results = vs.similarity_search("your query here", k=5)
for doc in results:
    print(doc.metadata, doc.page_content[:100])

# Load BM25
with open("index/faiss/bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)
with open("index/faiss/chunk_texts.pkl", "rb") as f:
    chunk_texts = pickle.load(f)

# Keyword search
query = "your query here"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)
top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:5]
for idx in top_indices:
    print(chunk_texts[idx])