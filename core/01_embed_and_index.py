"""
01_embed_and_index.py
Ngày 3–5: Load semantic chunks → embed bằng bge-m3 → build FAISS IndexFlatIP.

Chạy:
    python scripts/01_embed_and_index.py
    python scripts/01_embed_and_index.py --input /path/to/semantic_chunks.jsonl
    python scripts/01_embed_and_index.py --batch-size 16   # nếu RAM thấp
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "BAAI/bge-m3-small-v1.5"
model_kwargs = {"device": "cpu"} # hoặc "cuda" nếu có GPU
encode_kwargs = {"normalize_embeddings": True} # cần cho cosine via inner product

hf = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs)



