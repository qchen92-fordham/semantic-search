"""
CLI semantic search engine.

Run with: python semantic_search.py --query "your question"

1. Loads a text file of documents (blank-line separated)
2. Splits them into overlapping word chunks (default 120 words, 20-word overlap)
3. Embeds every chunk using the all-MiniLM-L6-v2 sentence transformer model
4. Builds a FAISS vector index for fast cosine similarity search
5. Takes your query, embeds it the same way, and prints the top-k most semantically similar chunks with their scores
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def load_documents(path: Path) -> List[str]:
    data = path.read_text(encoding="utf-8").strip()
    documents = [doc.strip() for doc in data.split("\n\n") if doc.strip()]
    return documents


def build_chunks(documents: List[str], chunk_size: int = 120, overlap: int = 20) -> List[Tuple[str, int, int]]:
    chunked = []
    for doc_i, doc in enumerate(documents):
        for chunk in chunk_text(doc, chunk_size=chunk_size, overlap=overlap):
            chunked.append((chunk, doc_i, len(chunk)))
    return chunked


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search(query: str, model: SentenceTransformer, index: faiss.IndexFlatIP, chunks: List[Tuple[str, int, int]], top_k: int = 5) -> List[Tuple[float, str]]:
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk_text, doc_i, _ = chunks[idx]
        results.append((float(score), chunk_text))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple semantic search with FAISS + embeddings")
    parser.add_argument("--source", type=str, default="docs/sample_faq.txt", help="Path to a text file with documents separated by blank lines")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--chunk-size", type=int, default=120, help="Chunk size in words")
    parser.add_argument("--overlap", type=int, default=20, help="Word overlap between chunks")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    documents = load_documents(source_path)
    chunks = build_chunks(documents, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise ValueError("No chunks generated from source documents.")

    print(f"Loaded {len(documents)} documents and generated {len(chunks)} chunks.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [text for text, _, _ in chunks]
    embeddings = embed_texts(model, texts)
    index = build_faiss_index(embeddings)

    if args.query:
        query_text = args.query
    else:
        query_text = input("Enter search query: ").strip()

    results = search(query_text, model, index, chunks, top_k=args.top_k)
    print("\nTop results:")
    for score, chunk in results:
        print(f"[{score:.4f}] {chunk}\n")


if __name__ == "__main__":
    main()
