import re
import requests
#from rank_bm25 import BM25Okapi
#from sentence_transformers import SentenceTransformer
#import numpy as np

#def ollama_llm(prompt: str, model: str = "llama3"):
#    """Placeholder Ollama client wrapper."""
#    return {"model": model, "prompt": prompt, "response": "Ollama placeholder response"}


# ---------------- LLM ----------------
def ollama_llm(prompt: str, model: str = "mistral") -> str:
    url = "http://localhost:11434/api/generate"
    response = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]