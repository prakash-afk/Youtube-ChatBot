from __future__ import annotations

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def merge_transcript_lines(transcript: list[dict]) -> list[dict]:
    merged: list[dict] = []
    buffer_text: list[str] = []
    start_time = 0.0
    end_time = 0.0

    for item in transcript:
        text = item.get("text", "").strip()
        if not text:
            continue

        item_start = float(item.get("start", 0.0))
        item_end = item_start + float(item.get("duration", 0.0))

        if not buffer_text:
            start_time = item_start

        buffer_text.append(text)
        end_time = item_end

        combined_text = " ".join(buffer_text)
        if len(combined_text) >= 180 or text.endswith((".", "?", "!")):
            merged.append(
                {
                    "text": combined_text,
                    "start": start_time,
                    "end": end_time,
                }
            )
            buffer_text = []

    if buffer_text:
        merged.append(
            {
                "text": " ".join(buffer_text),
                "start": start_time,
                "end": end_time,
            }
        )

    return merged


def build_documents(
    transcript: list[dict], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    merged_items = merge_transcript_lines(transcript)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for item in merged_items:
        parts = splitter.split_text(item["text"])
        for part in parts:
            documents.append(
                Document(
                    page_content=part,
                    metadata={
                        "start": item["start"],
                        "end": item["end"],
                    },
                )
            )

    return documents


def transcript_preview(transcript: list[dict], limit: int = 1500) -> str:
    text = " ".join(item.get("text", "").strip() for item in transcript if item.get("text"))
    return text[:limit]
