"""
Streamlit web UI for semantic search.

Run with: streamlit run app.py

Same pipeline as semantic_search.py but wrapped in a browser interface:
- Search box and sidebar sliders for chunk size, overlap, and number of results
- Caches the model and FAISS index in Streamlit's session cache to avoid recomputing on every keystroke
- Displays results as scored, formatted text blocks
"""

import streamlit as st
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    """Split a document into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def load_documents(path: Path) -> List[str]:
    """Load documents separated by blank lines from a text file."""
    data = path.read_text(encoding="utf-8").strip()
    return [doc.strip() for doc in data.split("\n\n") if doc.strip()]


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Encode a list of texts to normalized embeddings for cosine search."""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embeddings)
    return embeddings.astype(np.float32)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS index for inner-product similarity search."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search(query: str, model: SentenceTransformer, index: faiss.IndexFlatIP, chunks: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
    """Search the index and return top-k chunks with similarity scores."""
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    return [(float(distances[0][i]), chunks[indices[0][i]]) for i in range(len(indices[0])) if indices[0][i] >= 0]


def load_search_resources(path: Path, chunk_size: int = 120, overlap: int = 20):
    """Load documents, chunk them, encode embeddings, and build the FAISS index."""
    documents = load_documents(path)
    chunks = [chunk for doc in documents for chunk in chunk_text(doc, chunk_size=chunk_size, overlap=overlap)]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_texts(model, chunks)
    index = build_index(embeddings)
    return model, index, chunks


@st.cache_data(show_spinner=False)
def get_resources(source_path: str, chunk_size: int, overlap: int):
    """Cache model, index, and chunks in Streamlit to avoid recomputing on every rerun."""
    return load_search_resources(Path(source_path), chunk_size=chunk_size, overlap=overlap)


# Streamlit UI setup
st.title("Semantic Search Demo")
source_path = "docs/sample_faq.txt"
chunk_size = st.sidebar.slider("Chunk size (words)", min_value=50, max_value=250, value=120, step=10)
overlap = st.sidebar.slider("Chunk overlap (words)", min_value=0, max_value=80, value=20, step=5)

# Load resources once and reuse them for every search
model, index, chunks = get_resources(source_path, chunk_size, overlap)

# Text input for the semantic search query
query = st.text_input("Enter your search query", value="How do I reset my password?")
num_results = st.sidebar.slider("Top K results", min_value=1, max_value=10, value=5)

if query:
    results = search(query, model, index, chunks, top_k=num_results)
    st.markdown(f"### Top {len(results)} results")
    for score, chunk in results:
        st.write(f"**Score:** {score:.4f}")
        st.write(chunk)
        st.markdown("---")
