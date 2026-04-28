"""
Utilities cho Pipeline 2 - Cross-lingual Knowledge Units
Tac dung:
  - Xu li file JSONL (doc/ghi)
  - Trich xuat keywords (tieng Anh)
  - Dich keywords sang tieng Viet
  - Phat hien topic chinh

Su dung: config1.py va pipeline2.py goi cac ham trong file nay
"""

import re
import json
from pathlib import Path
from typing import Iterable


# TU KHOA TIENG ANH (AI/ML)
# Nhung tu/khoa thuong xuat hien trong bai giang AI/ML
# Dung de trich xuat keywords tu chunk_text
EN_KEYWORDS = (
    # AI/ML Core
    "neural",
    "network",
    "machine learning",
    "learning",
    "deep",
    "transformer",
    "attention",
    "backpropagation",
    "gradient",
    "optimization",
    "loss",
    "activation",
    "embedding",
    "convolutional",
    "recurrent",
    "vector",
    "tensor",
    # Models
    "gpt",
    "bert",
    "llm",
    "language model",
    "chatgpt",
    # Training
    "training",
    "test set",
    "validation",
    "overfitting",
    "regularization",
    "batch",
    "epoch",
    "learning rate",
    "hyperparameter",
    "fine-tuning",
    "fine tuning",
    # NLP
    "tokenizer",
    "tokenization",
    "vocabulary",
    "self-attention",
    "multi-head",
    "positional",
    # Concepts
    "softmax",
    "logit",
    "probability",
    "inference",
    "forward",
    "backward",
    # RL
    "reward",
    "policy",
    "agent",
    "action",
)


# TU KHOA TIENG VIET TƯƠNG ỨNG
# Dung de hien thi Vietnamese hints cho nguoi dung VN
VI_KEYWORDS = {
    "neural": "mang neural",
    "network": "mang",
    "machine": "may",
    "learning": "hoc",
    "deep": "sau",
    "transformer": "bien hinh",
    "attention": "chu y",
    "backpropagation": "truyen nguoc",
    "gradient": "dao ham",
    "optimization": "toi uu",
    "loss": "mat",
    "activation": "kich hoat",
    "embedding": "nhung",
    "convolutional": "tich chap",
    "recurrent": "hoi quy",
    "vector": "vec to",
    "tensor": "tensor",
    "gpt": "gpt",
    "bert": "bert",
    "llm": "llm",
    "training": "huan luyen",
    "validation": "kiem dinh",
    "overfitting": "qua khop",
    "regularization": "chinh quy",
    "batch": "luot",
    "epoch": "vong",
    "hyperparameter": "sieu tham so",
    "fine-tuning": "dieu chinh",
    "tokenizer": "ma hoa",
    "tokenization": "ma hoa",
    "vocabulary": "tu vung",
    "self-attention": "tu chu y",
    "multi-head": "nhieu dau",
    "positional": "vi tri",
    "softmax": "softmax",
    "logit": "logit",
    "probability": "xac suat",
    "inference": "suy luan",
    "forward": "truyen xuoi",
    "backward": "truyen nguoc",
    "reward": "thuong",
    "policy": "chinh sach",
    "agent": "dai ly",
    "action": "hanh dong",
    "token": "tu",
}


# STOPWORDS
# Cac tu thuong bi bo qua khi xu ly text
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "its",
    "as",
    "at",
    "by",
    "from",
    "if",
    "then",
    "than",
    "into",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "me",
    "our",
    "your",
    "their",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "not",
    "just",
    "there",
    "this",
    "these",
    "those",
    "here",
    "also",
    "now",
    "very",
    "more",
    "most",
}


# DOC FILE JSONL
# Doc tung dong tu file JSONL
# Yuan: Tung record mot luc de tiet kiem bo nho
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


# GHI FILE JSONL
# Ghi danh sach dict ra file JSONL
# Returns: So luong records da ghi
def write_jsonl(records: list, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# TRÍCH XUẤT KEYWORDS
# Tim cac tu khoa AI/ML trong text
# Args:
#   text: Van ban can xu ly
#   max_keywords: So tu khoa toi da (mac dinh: 8)
#   min_length: Do dai toi thieu cua tu khoa (mac dinh: 3)
# Returns:
#   List cac tu khoa tim duoc
def extract_keywords(
    text: str, max_keywords: int = 8, min_length: int = 3
) -> list[str]:
    """
    Trich xuat keywords tu text

    Args:
        text: Van ban can xu ly
        max_keywords: So tu khoa toi da
        min_length: Do dai toi thieu cua tu

    Returns:
        List cac tu khoa tim duoc (sap xep theo do dai)
    """
    # Chuyen ve chu thuong de tim kiem
    text_lower = text.lower()

    # Tim cac tu khoa trong EN_KEYWORDS co trong text
    found = []
    for kw in EN_KEYWORDS:
        if kw in text_lower:
            found.append(kw)

    # Loai bo tu qua ngan
    keywords = [kw for kw in found if len(kw) >= min_length]

    # Sap xep: uu tien tu dai hon (vi du: "machine learning" quan trong hon "learning")
    keywords.sort(key=len, reverse=True)

    return keywords[:max_keywords]


# DICH KEYWORDS SANG TIENG VIET
# Chuyen doi keywords tieng Anh sang tieng Viet
# Args:
#   en_keywords: List keywords tieng Anh
# Returns:
#   List keywords tieng Viet tuong ung
def get_vi_keywords(en_keywords: list[str]) -> list[str]:
    """
    Dich keywords sang tieng Viet

    Args:
        en_keywords: List keywords tieng Anh

    Returns:
        List keywords tieng Viet
    """
    vi_kw = []
    for en in en_keywords:
        if en in VI_KEYWORDS:
            vi_kw.append(VI_KEYWORDS[en])
    return vi_kw


# PHAT HIEN TOPIC
# Phat hien topic chinh cua chunk dua tren keywords
# Args:
#   text: Van ban can xu ly
# Returns:
#   Ten topic (vi du: "Neural Networks", "Training/Optimization")
def detect_topic(text: str) -> str:
    """
    Phat hien topic chinh cua chunk

    Args:
        text: Van ban can xu ly

    Returns:
        Ten topic chinh
    """
    text_lower = text.lower()

    # Topic mappings: keyword -> topic name
    topic_patterns = {
        "Neural Networks": ["neural network", "layer", "hidden", "weight", "bias"],
        "Transformer/Attention": [
            "transformer",
            "attention",
            "self-attention",
            "multi-head",
            "positional",
        ],
        "Training/Optimization": [
            "training",
            "learning rate",
            "optimizer",
            "gradient",
            "loss",
            "backprop",
        ],
        "NLP/Language Models": [
            "nlp",
            "language model",
            "token",
            "bert",
            "gpt",
            "embedding",
        ],
        "Computer Vision": ["image", "pixel", "convolution", "cnn", "feature"],
        "Reinforcement Learning": [
            "reward",
            "policy",
            "agent",
            "action",
            "environment",
        ],
    }

    # Dem so luot xuat hien cua moi topic
    scores = {}
    for topic, keywords in topic_patterns.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[topic] = score

    # Tra ve topic co diem cao nhat (neu > 0)
    if scores:
        best_topic = max(scores, key=scores.get)
        if scores[best_topic] > 0:
            return best_topic

    return "General AI/ML"


# DEM SO TU
# Dem so tu trong text
def word_count(text: str) -> int:
    return len(text.split())


# UOC LUONG TOKEN
# Uoc tinh so token (1 token ~ 1.3 words)
def token_count(text: str) -> int:
    return int(word_count(text) * 1.3)
