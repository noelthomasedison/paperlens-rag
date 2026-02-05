from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import re
import numpy as np
from pypdf import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
import re

@dataclass
class Chunk:
    doc_name: str
    page: int
    chunk_id: int
    text: str
    is_references: bool = False



def _clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_pdf_pages(file_bytes: bytes) -> List[str]:
    """Return list of page texts (one string per page)."""
    reader = PdfReader(io_bytes := _bytes_to_filelike(file_bytes))
    pages = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(_clean_text(txt))
    return pages


def _bytes_to_filelike(b: bytes):
    # small helper so pypdf can read bytes
    import io
    return io.BytesIO(b)


def chunk_pages(
    doc_name: str,
    pages: List[str],
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 250,
) -> List[Chunk]:
    """
    Chunk each page into overlapping character windows.
    Keeps page-level citation metadata.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    for i, page_text in enumerate(pages):
        page_num = i + 1
        if not page_text or len(page_text) < min_chars:
            continue
        start = 0
        while start < len(page_text):
            end = min(len(page_text), start + chunk_chars)
            text = page_text[start:end].strip()
            if len(text) >= min_chars:
                lower = page_text.lower()
                is_refs_page = ("references" in lower[:200] or "bibliography" in lower[:200])
                chunks.append(
                    Chunk(doc_name=doc_name, page=page_num, chunk_id=chunk_id, text=text, is_references=is_refs_page)
                )
                chunk_id += 1
            if end == len(page_text):
                break
            start = max(0, end - overlap_chars)

    return chunks


class RAGIndex:
    """
    Minimal FAISS cosine-similarity index over chunk embeddings.
    Stores metadata for citations.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.index: Optional[faiss.Index] = None
        self._emb_dim: Optional[int] = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # important for cosine via dot product
        )
        return vecs.astype("float32")

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index.")

        self.chunks = chunks
        vecs = self._embed([c.text for c in chunks])
        self._emb_dim = vecs.shape[1]

        self.index = faiss.IndexFlatIP(self._emb_dim)  # cosine via normalized dot product
        self.index.add(vecs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.index or not self.chunks:
            return []

        qv = self._embed([query])
        scores, ids = self.index.search(qv, top_k)
        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results


def format_citation(c: Chunk) -> str:
    return f"{c.doc_name} p.{c.page}"


def extractive_answer(question: str, retrieved: List[Tuple[Chunk, float]],include_headers: bool = False) -> str:
    """
    Improved extractive answer:
    - selects sentences relevant to the question
    - avoids token dumps and reference noise
    """
    print("DEBUG include_headers =", include_headers)

    if not retrieved:
        return "I couldn’t find relevant text in the uploaded PDFs."

    q_terms = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    
    def is_caption_or_header(sent: str) -> bool:
        s = sent.strip().lower()

        # figure / table captions anywhere in sentence
        if "figure" in s or "table" in s:
            return True

        # appendix headers (A.5, B.2, etc.)
        if re.match(r"^[a-z]\.\d+", s):
            return True

        # section headers like "3.1 Pre-training BERT"
        if re.match(r"^\d+(\.\d+)*\s+", s):
            return True

        # token soup / diagram labels
        if any(tok in s for tok in ["tokm", "mask lm", "nermnli", "squad"]):
            return True

        # very short / non-prose lines
        if len(s.split()) < 10:
            return True

        return False

    def score_sentence(sent: str) -> int:
        s = sent.lower()
        # ⛔ drop headers/captions unless toggle is ON
        if not include_headers and is_caption_or_header(sent):
            return -10_000

        # basic keyword overlap
        words = set(re.findall(r"[a-zA-Z]{4,}", s))
        score = len(words & q_terms)

        # boost definition-style sentences
        if any(k in s for k in ["pre-train", "pretraining", "objective", "task"]):
            score += 3
        if "bert" in s:
            score += 1

        # ✅ extra boost for "two tasks/objectives" type questions (BERT case)
        if any(t in question.lower() for t in ["two", "objectives", "tasks", "pre-training", "pretraining"]):
            if "masked" in s or "mask" in s or "mlm" in s:
                score += 3
            if "next sentence" in s or "nsp" in s or "prediction" in s:
                score += 3

        # penalize token junk and references
        if "[" in sent and "]" in sent:
            score -= 2
        if any(tok in s for tok in ["e[cls]", "e[sep]", "tokm", "##"]):
            score -= 4
        if re.search(r"\(\d{4}\)", sent):  # citation years
            score -= 1

        return score

    lines = []
    for chunk, _score in retrieved[:5]:
        sents = re.split(r"(?<=[.!?])\s+", chunk.text)
        sents = [s.strip() for s in sents if len(s.strip()) > 40]
        sents = [s for s in sents if len(s.split()) >= 10]
        # 🚫 HARD FILTER captions & headers when toggle is OFF
        if not include_headers:
            sents = [s for s in sents if not is_caption_or_header(s)]
        if not sents:
            continue

        ranked = sorted(sents, key=score_sentence, reverse=True)
        ranked = [s for s in ranked if score_sentence(s) > 0]
        picked = " ".join(ranked[:2]).strip()

        if picked:
            lines.append(f"- {picked} [{format_citation(chunk)}]")

    header = (
        "The paper may not explicitly state a concise answer in one sentence. "
        "Below is the most relevant supporting evidence:\n"
        )
    return header + "\n".join(lines)


def build_context_block(retrieved: List[Tuple[Chunk, float]], max_chars: int = 4000) -> Tuple[str, List[str]]:
    """
    Creates context text for LLM + list of citation labels in the same order.
    """
    blocks = []
    cites = []
    used = 0
    for chunk, _ in retrieved:
        cite = format_citation(chunk)
        snippet = chunk.text
        block = f"[SOURCE: {cite}]\n{snippet}\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        cites.append(cite)
        used += len(block)
    return "\n".join(blocks), cites


from typing import Optional, Tuple

def llm_answer_openai(
    question: str,
    retrieved: List[Tuple[Chunk, float]]) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns: (answer, status, detail)

    status ∈ {"OK", "NO_KEY", "QUOTA", "ERROR"}
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "NO_KEY", None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    context, _ = build_context_block(retrieved, max_chars=6000)
    if not context.strip():
        return "I couldn’t find relevant text in the uploaded PDFs.", "OK", None

    system = (
        "You are a careful assistant answering ONLY from the provided sources.\n"
        "If the answer isn't in the sources, say you couldn't find it.\n"
        "Always include citations in square brackets like [Paper.pdf p.3].\n"
        "Do not invent citations.\n"
    )

    user = f"""Question: {question}

Sources:
{context}

Write a concise answer with citations."""
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip(), "OK", None

    except Exception as e:
        msg = str(e).lower()

        if "insufficient_quota" in msg or "rate limit" in msg or "429" in msg:
            return None, "QUOTA", str(e)

        return None, "ERROR", str(e)

