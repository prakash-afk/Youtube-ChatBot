from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import torch


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _clear_broken_local_proxies() -> None:
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        value = os.environ.get(key, "")
        if "127.0.0.1:9" in value:
            os.environ.pop(key, None)


_clear_broken_local_proxies()


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
    top_k: int = int(os.getenv("TOP_K", "3"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "96"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    use_cuda: bool = torch.cuda.is_available()


def get_settings() -> Settings:
    return Settings()
