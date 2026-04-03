from __future__ import annotations

from pathlib import Path

import streamlit as st
from youtube_transcript_api import TranscriptsDisabled
from youtube_transcript_api._errors import VideoUnavailable

from app.chunking import build_documents, transcript_preview
from app.config import get_settings
from app.embed_store import build_retriever, build_vector_store, save_vector_store
from app.generator import answer_question, load_text_generator
from app.transcript import extract_video_id, fetch_transcript


@st.cache_resource(show_spinner=False)
def get_generator(model_path: str):
    return load_text_generator(model_path)


def render_sources(docs: list) -> None:
    if not docs:
        return

    st.subheader("Retrieved transcript chunks")
    for index, doc in enumerate(docs, start=1):
        start = float(doc.metadata.get("start", 0.0))
        end = float(doc.metadata.get("end", 0.0))
        st.markdown(f"**Chunk {index}** `{start:.1f}s - {end:.1f}s`")
        st.write(doc.page_content)


def main() -> None:
    settings = get_settings()
    Path(settings.faiss_index_dir).mkdir(parents=True, exist_ok=True)

    st.set_page_config(page_title="YouTube Chatbot", page_icon=">", layout="wide")
    st.title("YouTube Transcript Chatbot")
    st.caption("Fully local retrieval and answer generation")

    with st.sidebar:
        st.subheader("Configuration")
        st.write(f"Chat model path: `{settings.chat_model_path}`")
        st.write(f"Embedding model path: `{settings.embedding_model_path}`")
        st.write(f"FAISS index path: `{settings.faiss_index_dir}`")
        st.write(f"CUDA available: `{settings.use_cuda}`")
        st.write("Transcript retrieval and answer generation both stay on this machine.")
        if not settings.use_cuda:
            st.warning("PyTorch is running in CPU-only mode, so answer generation can be slow.")

    video_input = st.text_input("YouTube URL or video ID")
    question = st.text_area("Ask a question about the video")

    if st.button("Build transcript index", type="primary"):
        if not video_input.strip():
            st.error("Enter a YouTube URL or video ID first.")
        else:
            video_id = extract_video_id(video_input)
            if not video_id:
                st.error("Could not extract a valid YouTube video ID.")
            else:
                try:
                    with st.spinner("Fetching transcript and building the local index..."):
                        transcript = fetch_transcript(video_id)
                        documents = build_documents(
                            transcript,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap,
                        )
                        vector_store = build_vector_store(
                            documents,
                            embedding_model_path=settings.embedding_model_path,
                        )
                        save_vector_store(
                            vector_store,
                            index_dir=settings.faiss_index_dir,
                            video_id=video_id,
                        )
                        retriever = build_retriever(vector_store, settings.top_k)
                        st.session_state["vector_store"] = vector_store
                        st.session_state["retriever"] = retriever
                        st.session_state["video_id"] = video_id
                        st.session_state["transcript_preview"] = transcript_preview(transcript)
                    st.success("Transcript indexed successfully.")
                except TranscriptsDisabled:
                    st.error("Transcripts are disabled for this video.")
                except VideoUnavailable:
                    st.error("This video is unavailable or its transcript cannot be retrieved. Try opening the video in YouTube to confirm the ID.")
                except Exception as exc:
                    st.error(f"Could not build the transcript index: {exc}")

    if "transcript_preview" in st.session_state:
        with st.expander("Transcript preview"):
            st.write(st.session_state["transcript_preview"])

    if st.button("Ask"):
        if "retriever" not in st.session_state:
            st.error("Build the transcript index first.")
        elif not question.strip():
            st.error("Enter a question first.")
        elif not Path(settings.chat_model_path).exists():
            st.error("Local chat model path not found. Download the chat model first.")
        elif not Path(settings.embedding_model_path).exists():
            st.error("Local embedding model path not found. Download the embedding model first.")
        else:
            try:
                with st.spinner("Generating answer from the retrieved transcript context..."):
                    generator = get_generator(settings.chat_model_path)
                    answer, docs = answer_question(
                        generator,
                        st.session_state["retriever"],
                        question,
                        settings.max_context_chars,
                        settings.max_new_tokens,
                        settings.temperature,
                    )
                st.subheader("Answer")
                st.write(answer)
                render_sources(docs)
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")


if __name__ == "__main__":
    main()
