# YouTube Transcript Chatbot

A local-first Streamlit app that lets you ask questions about a YouTube video using its transcript.

The app:
- fetches the transcript from YouTube
- splits it into chunks
- creates embeddings with a local Hugging Face embedding model
- stores chunks in FAISS
- retrieves the most relevant transcript chunks for a question
- generates an answer with a local Hugging Face chat model

## Project Structure

```text
Youtube_CHATBOT/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── transcript.py
│   ├── chunking.py
│   ├── embed_store.py
│   ├── generator.py
│   └── main.py
└── data/
    └── faiss_index/
```

## Setup

Create and activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Download Local Models

Download the chat model:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct', local_dir='.hf_models/Qwen2.5-1.5B-Instruct', local_dir_use_symlinks=False)"
```

Download the embedding model:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='.hf_models/all-MiniLM-L6-v2', local_dir_use_symlinks=False)"
```

## Environment

Copy values from `.env.example` into your local `.env` if needed.

Current example settings:

```env
HF_CHAT_MODEL_PATH=.hf_models/Qwen2.5-1.5B-Instruct
EMBEDDING_MODEL_PATH=.hf_models/all-MiniLM-L6-v2
FAISS_INDEX_DIR=data/faiss_index
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=3
MAX_CONTEXT_CHARS=4000
MAX_NEW_TOKENS=96
TEMPERATURE=0.2
```

## Run

Start the app:

```powershell
python -m streamlit run app.py
```

Open:

`http://localhost:8501`

## How It Works

1. Enter a YouTube URL or video ID.
2. Click `Build transcript index`.
3. The app fetches the transcript and builds a FAISS index from transcript chunks.
4. Ask a question about the video.
5. The app retrieves relevant chunks and answers using the local chat model.

## Notes

- `.env`, `.venv`, and `.hf_models` are ignored by Git.
- FAISS index output is stored under `data/faiss_index/`.
- On CPU-only PyTorch, generation can be slow.
- For faster local inference, a CUDA-enabled PyTorch install is recommended.
