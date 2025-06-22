"""
Embed all chunks → FAISS index + metadata JSON
• Encoder: BAAI/bge-small-en-v1.5 (384-d)  — CPU-friendly
• Saves:
      vectordb/index.faiss
      vectordb/meta.json
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Config ─────────────────────────────────────────────────────────────────
CHUNK_DIR     = Path("chunks")
VECTORDB_DIR  = Path("vectordb")
INDEX_PATH    = VECTORDB_DIR / "index.faiss"
META_PATH     = VECTORDB_DIR / "meta.json"
MODEL_NAME    = "BAAI/bge-small-en-v1.5"

VECTORDB_DIR.mkdir(exist_ok=True)

# ─── Load chunks ────────────────────────────────────────────────────────────
texts, meta = [], []
for jsonl in CHUNK_DIR.glob("*.jsonl"):
    with jsonl.open(encoding="utf-8") as fin:
        for ln in fin:
            obj = json.loads(ln)
            texts.append(obj["text"])
            meta.append(obj)           # keep id / file / page / text

if not texts:
    raise RuntimeError("No chunks found — run chunker.py first.")

# ─── Embed ──────────────────────────────────────────────────────────────────
print("⏳ loading BGE-small encoder …")
encoder = SentenceTransformer(MODEL_NAME, device="cpu")

print("⏳ encoding chunks …")
emb = encoder.encode(texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)\
                     .astype("float32")

# ─── Build / save index & metadata ──────────────────────────────────────────
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, str(INDEX_PATH))
META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

print(f"✅ vectordb built: {index.ntotal} vectors, dim {emb.shape[1]}")
