import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    
    if not text or not text.strip():
        return []

    # ── 1. Normalize whitespace ───────────────────────────────────────────────
    text = text.strip()
    text = re.sub(r'\n+', ' ', text)        # newlines to spaces
    text = re.sub(r'\s+', ' ', text)        # multiple spaces to one

    # ── 2. Protect known abbreviations from being split ───────────────────────
    # Replace period in abbreviation with a placeholder
    abbreviations = [
        r'Fig\.', r'fig\.', r'et al\.', r'vs\.', r'e\.g\.', r'i\.e\.',
        r'Dr\.', r'Prof\.', r'Mr\.', r'Mrs\.', r'Ms\.', r'Sr\.', r'Jr\.',
        r'approx\.', r'ref\.', r'no\.',
    ]
    placeholders = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"ABBR{i}PLACEHOLDER"
        text = re.sub(abbr, placeholder, text, flags=re.IGNORECASE)
        placeholders[placeholder] = re.sub(r'\\', '', abbr)

    # ── 3. Protect decimal numbers (3.14, 97.2%) ─────────────────────────────
    text = re.sub(r'(\d)\.(\d)', r'\1DECIMAL\2', text)

    # ── 4. Split on sentence-ending punctuation ───────────────────────────────
    sentences_raw = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text)

    # ── 5. Also handle bullet/numbered lists as separate sentences ────────────
    sentences = []
    for s in sentences_raw:
        # Split on bullet points
        parts = re.split(r'\s*[•\-]\s+', s)
        # Split on numbered list items (1. 2. etc)
        parts2 = []
        for p in parts:
            sub = re.split(r'\s+\d+\.\s+', p)
            parts2.extend(sub)
        sentences.extend(parts2)

    # ── 6. Restore placeholders ───────────────────────────────────────────────
    cleaned = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Restore decimals
        s = s.replace('DECIMAL', '.')
        # Restore abbreviations
        for placeholder, original in placeholders.items():
            s = s.replace(placeholder, original.replace('\\', ''))
        # Only keep sentences with at least 5 words (skip fragments)
        if len(s.split()) >= 5:
            cleaned.append(s)

    return cleaned


def preview_sentences(answer: str, max_chars: int = 80) -> None:
    sentences = split_into_sentences(answer)
    print(f"\n[splitter] Split into {len(sentences)} sentences:")
    for i, s in enumerate(sentences, 1):
        preview = s[:max_chars] + ('...' if len(s) > max_chars else '')
        print(f"  [{i}] {preview}")
    return sentences