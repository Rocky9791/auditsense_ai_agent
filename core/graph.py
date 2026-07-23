import re
import requests
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

from ingest import build_index, CORPUS
from hybrid_search import HybridSearcher
from reranker import Reranker

# CHANGED: removed `from click import prompt` — unused dead import, same smell
# as the urllib.response issue from earlier in the project.


def ollama_llm(prompt: str, model: str = "mistral") -> str:
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
    return response.json()["response"]


def claude_llm(prompt: str, model: str = "claude-sonnet-5") -> str:
    # CHANGED: NEW — same signature as ollama_llm(prompt) -> str, so build_graph
    # doesn't need to know or care which provider it's calling.
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


class AgentState(TypedDict):
    original_query: str
    is_multi_intent: bool
    sub_queries: list[str]
    retrieved_chunks: dict[str, list[str]]
    top_scores: dict[str, float]
    final_answer: str


def single_query_passthrough(state: AgentState) -> dict:
    # unchanged — doesn't touch the LLM, stays module-level
    return {"sub_queries": [state["original_query"]]}


def route_after_classify(state: AgentState) -> Literal["decompose_query", "single_query_passthrough"]:
    # unchanged — doesn't touch the LLM, stays module-level
    return "decompose_query" if state["is_multi_intent"] else "single_query_passthrough"


def build_graph(searcher, reranker, llm_call_fn=ollama_llm):
    """
    CHANGED: added `llm_call_fn` parameter (defaults to ollama_llm).
    classify_intent, decompose_query, and synthesize_answer are now closures
    defined INSIDE build_graph, so they call whichever llm_call_fn was passed in
    — same closure pattern you already used for retrieve_for_each with
    searcher/reranker. To demo Claude instead of Ollama:
        build_graph(searcher, reranker, llm_call_fn=claude_llm)
    """

    def classify_intent(state: AgentState) -> dict:
        query = state["original_query"].strip()
        result = llm_call_fn(
            f"Classify the following query as SINGLE or MULTI intent: {query} "
            "Strictly respond with SINGLE or MULTI only. One word response, nothing else."
        )
        return {"is_multi_intent": result.strip().upper() == "MULTI"}

    def decompose_query(state: AgentState) -> dict:
        if not state["is_multi_intent"]:
            raise ValueError("decompose_query should only be called for multi-intent queries.")

        query = state["original_query"].strip()
        result = llm_call_fn(
            f"Decompose the following multi-intent query into separate, standalone sub-questions: {query} "
            "Respond with a numbered list, one question per line. No other text."
        )
        print("\nRAW LLM OUTPUT:\n", result)

        queries = []
        for line in result.split("\n"):
            match = re.match(r"^\d+\.\s*(.+)", line.strip())
            if match:
                queries.append(match.group(1).strip())

        return {"sub_queries": queries}

    def retrieve_for_each(state: AgentState) -> dict:
        retrieved_chunks = {}
        top_scores = {}
        for sub_query in state["sub_queries"]:
            hybrid_results = searcher.hybrid_search(sub_query, top_k=6)
            candidates = [text for text, idx in hybrid_results]

            # CHANGED: removed the duplicate manual reranker.model.predict() call —
            # it computed cross-encoder scores a SECOND time purely for a debug print,
            # doubling reranking latency/compute for no functional benefit.
            # reranker.rerank() already computes these scores internally.
            reranked = reranker.rerank(sub_query, candidates, top_k=4)

            retrieved_chunks[sub_query] = [text for text, score in reranked]
            top_scores[sub_query] = reranked[0][1] if reranked else float("-inf")

        # CHANGED (BUG FIX #2): top_scores was computed but never returned before —
        # state["top_scores"] would have stayed permanently empty, causing a
        # guaranteed KeyError in synthesize_answer on every real run.
        return {"retrieved_chunks": retrieved_chunks, "top_scores": top_scores}

    def synthesize_answer(state: AgentState) -> dict:
        per_question_answers = []
        CONFIDENCE_THRESHOLD = 0.5  # TODO: replace with your calibrated value

        for sq in state["sub_queries"]:
            score = state["top_scores"][sq]
            # CHANGED (BUG FIX #1): answer-assignment and the ollama_llm call are now
            # correctly nested INSIDE the if/else — previously they sat outside both
            # branches (same indent as if/else itself), which meant:
            #   - low-confidence path crashed with UnboundLocalError (single_prompt
            #     was never defined on that branch), so the governance/confidence
            #     feature didn't just fail silently, it broke every request.
            #   - high-confidence path silently called the LLM a second redundant time.
            if score < CONFIDENCE_THRESHOLD:
                answer = "I don't have enough reliable information to answer this specific question."
            else:
                context = "\n".join(state["retrieved_chunks"][sq])
                single_prompt = (
                    f"Answer this question using ONLY the context below. "
                    f"If the context doesn't contain enough information, say so explicitly.\n\n"
                    f"Context:\n{context}\n\nQuestion: {sq}\nAnswer:"
                )
                answer = llm_call_fn(single_prompt)

            per_question_answers.append((sq, answer))

        if len(per_question_answers) == 1:
            return {"final_answer":per_question_answers[0][1]}
        
        elif not per_question_answers:
            return {"final_answer": "I wasn't able to identify a clear question to answer. Could you rephrase your query?"}
        
        else :
            combine_prompt = "Combine these Q&A pairs into one coherent, well-organized answer:\n\n"
            for sq, ans in per_question_answers:
                combine_prompt += f"Q: {sq}\nA: {ans}\n\n"

        return {"final_answer": llm_call_fn(combine_prompt)}

    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("single_query_passthrough", single_query_passthrough)
    graph.add_node("retrieve_for_each", retrieve_for_each)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"decompose_query": "decompose_query", "single_query_passthrough": "single_query_passthrough"}
    )
    graph.add_edge("decompose_query", "retrieve_for_each")
    graph.add_edge("single_query_passthrough", "retrieve_for_each")
    graph.add_edge("retrieve_for_each", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile()


if __name__ == "__main__":
    searcher, chunks, sources = build_index(CORPUS[:5], chunk_size=300, overlap=50)
    reranker = Reranker()

    # CHANGED: swap llm_call_fn=claude_llm here to demo the Claude backend instead
    agent = build_graph(searcher, reranker, llm_call_fn=ollama_llm)

    result = agent.invoke({
        "original_query": "does the company reimburse cryptocurrency trading losses?",
        "is_multi_intent": False,
        "sub_queries": [],
        "retrieved_chunks": {},
        "top_scores": {},
        "final_answer": ""
    })

    print("SUB-QUERIES this check yeh hai:", result["sub_queries"])
    print("\nFINAL ANSWER:\n", result["final_answer"])