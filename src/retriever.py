"""
Lightweight retriever
• FAISS cosine search on BGE-small embeddings
• keyword-ratio filter keeps recall high, junk low
"""

from __future__ import annotations

import json, re, string
from pathlib import Path
from typing import List, TypedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── config ────────────────────────────────────────────────────────────────
VECTORDB_DIR  = Path("vectordb")
INDEX_PATH    = VECTORDB_DIR / "index.faiss"
META_PATH     = VECTORDB_DIR / "meta.json"
EMB_MODEL     = "BAAI/bge-small-en-v1.5"

SIM_THRESHOLD = 0.45         # was 0.60 – empirically safer
MAX_K_HITS    = 6            # default k for each search
RATIO_MIN     = 0.40         # ≥ 40 % keyword overlap

_STOP    = {"the", "a", "an", "of", "and", "to", "for", "is", "are", "in", "on"}
_IGNORE  = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}


class Doc(TypedDict):
    id: str
    file: str
    page: int
    text: str
    score: float


class Retriever:
    def __init__(self, default_k: int = MAX_K_HITS) -> None:
        self.default_k = default_k
        self._index = faiss.read_index(str(INDEX_PATH))
        self._meta: List[dict] = json.loads(META_PATH.read_text(encoding="utf-8"))
        self._enc = SentenceTransformer(EMB_MODEL, device="cpu")

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _keywords(txt: str) -> set[str]:
        return {
            t.lower().strip(string.punctuation)
            for t in re.split(r"\W+", txt)
            if t
            and len(t) > 2
            and t.lower() not in _STOP
            and t.lower() not in _IGNORE
        }

    def search(self, query: str, k: int = MAX_K_HITS) -> List[Doc]:
        """
        Return up to *k* semantically similar passages whose
        cosine-sim ≥ SIM_THRESHOLD and keyword overlap ≥ RATIO_MIN.
        """
        # encode & query FAISS
        q_vec = (
            self._enc.encode([query], normalize_embeddings=True)
            .astype("float32")
        )
        scores, ids = self._index.search(q_vec, k)

        # cosine similarity filter
        cand = [
            (i, s)
            for i, s in zip(ids[0], scores[0])
            if i != -1 and s >= SIM_THRESHOLD
        ]

        # keyword-ratio sanity check
        q_keys = self._keywords(query)

        def ratio(text: str) -> float:
            if not q_keys:
                return 1.0
            w = self._keywords(text)
            return len(q_keys & w) / len(q_keys)

        hits: List[Doc] = []
        for idx, score in cand:
            meta = self._meta[int(idx)]
            if ratio(meta["text"]) >= RATIO_MIN:
                hits.append({**meta, "score": float(score)})

        # sort highest-to-lowest score (FAISS is already sorted but
        # secondary filters can disturb order)
        return sorted(hits, key=lambda d: d["score"], reverse=True)


# # quick sanity check (delete in production) ----------------------------
# if __name__ == "__main__":
#     r = Retriever()
#     for doc in r.search("why did the project timeline slip?", k=4):
#         print(f"{doc['score']:.3f}", doc["text"][:120].replace("\n", " "), "…")
