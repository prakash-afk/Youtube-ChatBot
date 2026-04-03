from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    chat_model_path: str = os.getenv(
        "HF_CHAT_MODEL_PATH", str(BASE_DIR / ".hf_models" / "Qwen2.5-1.5B-Instruct")
    )
    embedding_model_path: str = os.getenv(
        "EMBEDDING_MODEL_PATH", str(BASE_DIR / ".hf_models" / "all-MiniLM-L6-v2")
    )
    faiss_index_dir: str = os.getenv(
        "FAISS_INDEX_DIR", str(BASE_DIR / "data" / "faiss_index")
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))


def get_settings() -> Settings:
    return Settings()
