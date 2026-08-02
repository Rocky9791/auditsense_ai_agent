
import os

from flask import Flask, request, jsonify
from core.ingest import build_index, CORPUS
from core.reranker import Reranker
from core.graph import build_graph

app = Flask(__name__)

# Build the index ONCE at startup — same principle as your main.py before
searcher = None
reranker = None
agent = None

@app.route("/query", methods=["POST"])
def query():

    global searcher, reranker, agent

    if agent is None:
        searcher, chunks, sources = build_index(CORPUS[:5], chunk_size=300, overlap=50)
        #reranker = Reranker()
        agent = build_graph(searcher, reranker)


    data = request.get_json()
    user_query = data.get("query", "").strip()
    
    if not user_query:
        return jsonify({"error": "query field is required"}), 400

    result = agent.invoke({
        "original_query": user_query,
        "is_multi_intent": False,
        "sub_queries": [],
        "retrieved_chunks": {},
        "top_scores": {},
        "final_answer": ""
    })

    return jsonify({
        "query": user_query,
        "sub_queries": result["sub_queries"],
        "answer": result["final_answer"]
    })

#if __name__ == "__main__":
#   app.run(debug=True, port=5000)

#if __name__ == "__main__":
 #   port = int(os.environ.get("PORT", 10000))
  #  app.run(host="0.0.0.0", port=port)