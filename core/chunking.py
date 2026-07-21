
import re

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    text = text.strip()
    n = len(text)
    start = 0

    while start < n:
        max_end = min(start + chunk_size, n)
        window = text[start:max_end]

        match = list(re.finditer(r'[.!?]', window))

        if match:
            split_idx = match[-1].end()
            end = start + split_idx
        else:
            end = max_end

      
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # KEY FIX: if we've reached the end of the document, stop — no more chunks needed
        if end >= n:
            break

        #  FIX: enforce forward movement
        new_start = end - overlap
        #  WORD-BOUNDARY FIX
        while new_start > 0 and text[new_start] not in [" ", "\n"]:
            new_start -= 1

        if new_start <= start:
            new_start = start + 1   # force progress
        

        # Ensure forward movement
        if new_start <= start:
            new_start = start + 1

        start = new_start

    return chunks
