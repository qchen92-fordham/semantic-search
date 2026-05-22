# Semantic Search Project

A simple semantic search demo using embeddings, FAISS vector search, and chunking.

## What it does

- Loads sample text documents
- Splits text into chunks with overlap
- Encodes chunks using `sentence-transformers`
- Builds a FAISS vector index using cosine similarity
- Retrieves top-k text chunks for a query

## Files

- `semantic_search.py` - core search pipeline
- `app.py` - optional Streamlit UI
- `docs/sample_faq.txt` - sample documents for search
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run CLI search

```bash
python semantic_search.py --query "where is the capital of Cataluña"
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## Resume-friendly features

- built a semantic search engine using embeddings + FAISS
- implemented chunking strategy for long text documents
- performed approximate nearest neighbor search with cosine similarity
- returned ranked results and similarity scores
- optional UI using Streamlit for interactive search
