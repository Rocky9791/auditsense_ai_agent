
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridSearcher:
    def __init__(self, documents: list[str]):
        self.documents = documents
        
        # BM25 setup
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Embeddings
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_embeddings = self.embed_model.encode(documents, normalize_embeddings=True)

    def keyword_search(self, query: str, top_k: int = 10) -> list[int]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = np.argsort(scores)[::-1]  # descending
        return ranked_indices[:top_k].tolist()

    def semantic_search(self, query: str, top_k: int = 10) -> list[int]:
        query_embedding = self.embed_model.encode(query, normalize_embeddings=True)

        similarities = np.dot(self.doc_embeddings, query_embedding)
        ranked_indices = np.argsort(similarities)[::-1]

        return ranked_indices[:top_k].tolist()

    def reciprocal_rank_fusion(self, ranked_lists: list[list[int]], k: int = 60) -> list[int]:
        scores = {}

        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                # rank is 0-indexed → convert to 1-indexed
                r = rank + 1
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + r)

        # sort by fused score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in sorted_docs]

    def hybrid_search(self, query: str, top_k: int = 5) -> list[tuple[str, int]]:
        keyword_rank = self.keyword_search(query, top_k=top_k)
        semantic_rank = self.semantic_search(query, top_k=top_k)

        fused = self.reciprocal_rank_fusion([keyword_rank, semantic_rank])

        return [(self.documents[i], i) for i in fused[:top_k]]
