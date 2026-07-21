from chunking import chunk_text
from hybrid_search import HybridSearcher

def extract_header_from_text(doc):
    return doc[:60]

def build_index(corpus: list[str], chunk_size: int, overlap: int):
    """
    TODO:
    1. Run each document in `corpus` through chunk_text().
    2. Collect ALL resulting chunks into a single flat list (this is what 
       HybridSearcher will be built on).
    3. ALSO build a mapping so you can trace each chunk back to which 
       original document index it came from — you'll want this for citations 
       later. A parallel list `chunk_to_source_doc: list[int]` works fine: 
       chunk_to_source_doc[i] = index of the CORPUS entry that produced 
       all_chunks[i].
    4. Build and return a HybridSearcher over all_chunks (not over the raw corpus).
    
    Return: (searcher, all_chunks, chunk_to_source_doc)
    """
    
    all_chunks = []
    chunk_to_source_doc = []

    for i, doc in enumerate(corpus):
        if not isinstance(doc, str):
            raise ValueError("All documents in corpus must be strings")         
        
        chunks = chunk_text(doc, chunk_size = chunk_size, overlap = overlap)
        header = extract_header_from_text(doc)

        for chunk in chunks:
            all_chunks.append(f"{header}: {chunk}")
            chunk_to_source_doc.append(i)

    """BM25 works better on smaller text
    Embeddings are more precise on chunks , so not using corpus or docs to fetch searcher"""
    
    for i,text in enumerate(all_chunks):
        # print the chunk with its source document index mapped to document number for citation 
        print(f"Chunk {i+1} (from doc {chunk_to_source_doc[i]}): {text}")


    
    searcher = HybridSearcher(all_chunks) # just searcher instant is created here so chill bro 

    return searcher, all_chunks, chunk_to_source_doc




# if __name__ == "__main__":
    #from ingest import CORPUS  # or define CORPUS in this file directly
   # searcher, chunks, sources = build_index(CORPUS,chunk_size=300, overlap=50)
   # print(f"Corpus: {len(CORPUS)} documents → {len(chunks)} chunks")
    
    # Sanity check: run one real query and confirm chunk-level retrieval works
   # results = searcher.hybrid_search("does the phonex pro charge faster than the galaxy z", top_k=3)
    #for text, idx in results:
    #    print(f"[chunk {idx} ← doc {sources[idx]}] {text[:100]}...")


CORPUS = [
    # Expense policy
    "Employee travel expenses for domestic flights are reimbursable up to INR 15,000 per trip when booked at least 7 days in advance. Bookings made within 7 days require manager pre-approval and are capped at INR 20,000.",
    "Per-diem for domestic business travel is INR 2,500 per day, covering meals and incidentals. International per-diem varies by country and is listed in Appendix B of the travel policy.",
    "Hotel accommodation is reimbursable up to INR 8,000 per night in metro cities (Mumbai, Delhi, Bangalore, Chennai) and INR 5,000 per night in non-metro locations. Any amount exceeding this cap requires written justification and CFO approval.",
    "Client entertainment expenses require an itemized receipt and the names of all attendees. Expenses without attendee names on file will be flagged during audit review and may be rejected.",
    "Expense reports must be submitted within 30 days of the expense date. Reports submitted after 30 days require additional approval from the Finance Controller and a written explanation for the delay.",

    # Invoice / vendor policy
    "All vendor invoices above INR 100,000 require a Purchase Order (PO) number for processing. Invoices without a matching PO will be held in a pending-review queue and not paid until reconciled.",
    "Vendor payment terms are Net-30 by default. Any vendor requesting Net-15 or shorter terms requires approval from the Procurement Head and must be documented in the vendor master record.",
    "Duplicate invoice detection flags any invoice with the same vendor, amount, and date as a previously processed invoice within a 90-day window for manual review before payment release.",

    # Audit trail / compliance
    "All financial transactions above INR 50,000 must have a documented audit trail including requester, approver, and date of approval, retained for a minimum of 7 years per company records-retention policy.",
    "Segregation of duties requires that the individual who initiates a payment request cannot be the same individual who approves it. Violations of this control are flagged automatically during quarterly internal audits.",
    "Any change to vendor banking details must be verified via a callback to a previously on-file phone number before the change is processed, to prevent payment redirection fraud.",
    "Internal audit reviews are conducted quarterly and cover a random sample of at least 5% of transactions above INR 25,000 from the prior quarter.",

    # Reporting / month-end close
    "Month-end close activities must be completed within 5 business days of month-end, including account reconciliations, accruals, and variance analysis against budget.",
    "Any budget variance exceeding 10% for a cost center requires a written explanation from the cost center owner as part of the month-end reporting package.",

    # Data quality / governance — directly relevant to confidence_check framing
    "All AI-assisted compliance determinations must be reviewed by a human analyst when the system's confidence in its retrieved policy match falls below the defined threshold; such cases are logged and routed to the Finance Controls team queue.",
    "Data quality checks on financial datasets include validation of currency consistency, duplicate transaction detection, and completeness checks on required approval fields before any automated processing occurs.",
]

