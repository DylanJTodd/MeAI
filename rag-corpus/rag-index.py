## GENERATES RAG INDEX FROM RAG-DOCS.TXT

from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("rag_docs.txt", "r", encoding="utf-8") as f:
    docs = f.read().split("\n---\n")

embeddings = embedder.encode(docs, convert_to_tensor=False)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "rag-corpus/rag-index.faiss")
