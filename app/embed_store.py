from __future__ import annotations

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(documents, embedding_model_path: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)
    return FAISS.from_documents(documents, embedding=embeddings)


def save_vector_store(vector_store: FAISS, index_dir: str, video_id: str) -> str:
    target_dir = Path(index_dir) / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_dir))
    return str(target_dir)
