
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# -------------------------------
# Simple Bi-Encoder Vector Store
# -------------------------------
class SimpleVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list[str]):
        self.documents = documents
        self.embeddings = self.model.encode(documents, convert_to_numpy=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)

        # cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.documents[i], float(scores[i])) for i in top_indices]


# -------------------------------
# Cross-Encoder Reranker
# -------------------------------
class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str], top_k: int = 3) -> list[tuple[str, float]]:
        if not candidates:
            return []

        # 1. Build (query, candidate) pairs
        pairs = [(query, candidate) for candidate in candidates]

        # 2. Predict scores (batch for efficiency)
        scores = self.model.predict(pairs, batch_size=16)

        scores = self.model.predict(pairs)
        probs = 1 / (1 + np.exp(-scores))  # sigmoid

        # 3. Combine text + score
        scored_candidates = list(zip(candidates, probs))


        # 4. Sort descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 5. Return top_k
        return scored_candidates[:top_k]