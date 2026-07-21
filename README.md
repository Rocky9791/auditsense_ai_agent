Confidence threshold, calibrated on the PhoneSense corpus, transferred without adjustment to an unrelated finance domain and correctly gated an out-of-scope query"

# AuditSense — Agentic RAG for Finance Compliance & Audit Q&A

An agentic RAG system that answers questions about expense policy, vendor/invoice rules, and internal audit controls — with a governance-gating layer that escalates rather than guesses when retrieval confidence is low.

Built as a domain adaptation of [PhoneSense](#) (a consumer-product support agent) onto finance/compliance data, to test whether the same architecture generalizes without redesign.

## Problem

Naive RAG systems answer confidently even when they haven't actually found relevant information — a serious liability in finance/compliance contexts, where an incorrect policy determination has real consequences. AuditSense adds an explicit **confidence-gating control**: if the system's retrieval confidence for a sub-question falls below a calibrated threshold, it declines to answer that part rather than fabricating a plausible-sounding response, and signals that the case should be escalated for human review.

## Architecture

```
User query
   │
   ▼
classify_intent ──(LLM: single vs multi-intent)
   │
   ├── multi ──► decompose_query ──┐
   └── single ─► passthrough ──────┤
                                    ▼
                    retrieve_for_each  (hybrid search: BM25 + dense embeddings, fused via
                                         Reciprocal Rank Fusion → cross-encoder reranking)
                                    │
                                    ▼
                    synthesize_answer  (per sub-question: if confidence < threshold →
                                         explicit "insufficient information" fallback;
                                         else → grounded LLM answer from retrieved context)
                                    │
                                    ▼
                              Final answer
```

Deployed as a live Flask API (`POST /query`), with a swappable LLM backend (local Ollama by default, Anthropic Claude API optional).

## Key design decisions & evidence

| Decision | Why |
|---|---|
| Hybrid search (BM25 + embeddings via RRF), not embeddings alone | Keyword search catches exact terms (PO numbers, IP ratings, specific policy figures) that dense embeddings can under-weight; semantic search catches paraphrases keyword search misses. RRF fuses both without needing to normalize incompatible score scales. |
| Cross-encoder reranking as a second pass | Bi-encoder retrieval embeds query and document independently (fast, precomputable); a cross-encoder scores the query and document jointly (more accurate, but only feasible on a narrowed candidate set, since it can't be precomputed at index time). |
| Contextual chunk headers | Chunking a document can sever a sentence from the sentence that names its subject (coreference loss) — e.g., "It supports 45W charging" loses which product "it" refers to once split into a separate chunk. Prepending a short source-document header to every chunk preserves this. |
| Per-question-then-combine synthesis | A single large multi-part prompt did not reliably guarantee every sub-question got addressed — the LLM would sometimes silently omit the part it was least confident about. Answering each sub-question independently, then combining, makes omission structurally impossible. |
| Sigmoid-normalized confidence scores + calibrated threshold | Raw cross-encoder outputs are unbounded logits, not a 0–1 confidence scale. Applying sigmoid produces an interpretable probability, calibrated against real in-scope vs. out-of-scope test queries rather than picked arbitrarily. |

## Bugs found & fixed during development (evidence of real debugging, not just assembly)

1. **Chunk-boundary coreference loss** — a product/entity name in a chunk's opening sentence was lost when later sentences referencing it (via pronouns) landed in a separate chunk, causing retrieval to miss the correct chunk entirely. Fixed via contextual chunk headers.
2. **Synthesis silently dropping sub-questions** — a single combined prompt didn't reliably cover every sub-question under a local open-weight model. Fixed via per-question-then-combine architecture.
3. **Reranker narrowly demoting the correct chunk** — character-based chunk overlap could cut a chunk mid-word, degrading its embedding/rerank quality; a garbled chunk scored just below a topically-adjacent-but-irrelevant one. Mitigated by widening the retrieved candidate pool before reranking.
4. **Score-scale mismatch between independently-evolved files** — reranker output was upgraded to sigmoid-normalized probabilities, but the confidence threshold elsewhere in the codebase was still calibrated for the old raw-logit scale, silently breaking the gate. Fixed by recalibrating the threshold to match the new scale.
5. **Ungrounded hallucination via an unnecessary "combine" pass** — for single-question queries, a downstream LLM call meant to "combine multiple answers" had nothing real to combine, so it improvised — on one out-of-scope test query (asking about a policy area absent from the corpus), this caused the model to fabricate a plausible-sounding but ungrounded answer instead of returning the correctly-generated fallback string. Fixed by skipping the combine step entirely for single-question queries.
6. **Cross-domain generalization check** — the confidence threshold, originally calibrated on a consumer-electronics corpus, was tested unmodified against this unrelated finance corpus. It correctly separated a genuinely out-of-scope query (score ≈ 0.00002) from real in-scope retrievals, without any recalibration — evidence the gating design is robust rather than overfit to one dataset.

## Tech stack

Python · LangGraph · sentence-transformers (bi-encoder + cross-encoder) · rank_bm25 · Flask · Ollama (local LLM) / Anthropic Claude API (optional)

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python app.py
```

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the hotel reimbursement limit in Mumbai and does an invoice over 1 lakh need a PO number?"}'
```

## Related project

[PhoneSense](#) — the same architecture applied to consumer-product support Q&A, where this design was originally built and debugged.