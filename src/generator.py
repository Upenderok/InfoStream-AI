"""
Phi-3 mini wrapper for document-grounded RAG
-------------------------------------------
• deterministic (temperature 0.0)
• 3072-token context window
• strips any prompt leakage so Streamlit shows only the answer
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Generator, Union

from llama_cpp import Llama  # type: ignore

# ─── model & prompt ─────────────────────────────────────────────────────────

MODEL_PATH = Path("models/Phi-3-mini-4k-instruct-q4.gguf").as_posix()

PROMPT = """### System
You are an expert factual Q & A assistant that must **answer ONLY from the given CONTEXT**.
If the answer is not 100 % contained in CONTEXT, reply exactly: **I don’t know**.
Never fabricate, speculate, or combine outside knowledge.

* Cite every supporting fact with its chunk number in square brackets, e.g. [2].  
* If multiple chunks support the same sentence, list them comma-separated: [1, 4].  
* Do **not** invent chunk numbers.  
* Keep the answer concise but complete (≤ 250 words).  
* Use Markdown for formatting.  
* End on a new line with **Confidence: High | Medium | Low**.

### Context
{context}

### Question
{question}

### Answer (markdown):
"""

# Stop if the model starts echoing any of these
_STOP_SEQS = [
    "</s>",
    "### System",
    "### Context",
    "### Question",
    "### Answer",
    "Instruction",
    "instruction",
]

# ─── singleton loader ───────────────────────────────────────────────────────
_llm: Llama | None = None


def _load_llm() -> Llama:
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=3072,
            n_batch=64,
            n_threads=os.cpu_count(),
            mmap=True,
            n_gpu_layers=0,          # CPU-only
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            stop=_STOP_SEQS,
        )
    return _llm


# ─── prompt-leak scrubber (non-streaming fallback) ─────────────────────────
def _strip_prompt_leak(text: str) -> str:
    """Remove everything from the first leaked prompt heading onward."""
    return re.split(
        r"\n\s*(instruction|###|sources used).*",
        text,
        flags=re.IGNORECASE,
        maxsplit=1,
    )[0].rstrip()


# ─── public API ─────────────────────────────────────────────────────────────
def generate(
    context: str,
    question: str,
    *,
    stream: bool = True,
) -> Union[str, Generator[str, None, None]]:
    """
    Generate an answer grounded in *context*.

    Parameters
    ----------
    context : str
        Retrieved chunks (already numbered “[0] …”).
    question : str
        User query.
    stream : bool, default True
        If True, yields plain-string tokens; else returns one full string.
    """
    prompt = PROMPT.format(context=context, question=question)
    raw = _load_llm()(
        prompt,
        max_tokens=512,
        stream=stream,
    )

    # ── streaming mode ─────────────────────────────────────────────────────
    if stream:

        def _clean_gen() -> Generator[str, None, None]:
            buf = ""
            for chunk in raw:  # chunk is a dict from llama-cpp
                tok = chunk["choices"][0]["text"]
                buf += tok
                if any(stop.lower() in buf.lower() for stop in _STOP_SEQS):
                    break
                yield tok

        return _clean_gen()

    # ── non-streaming mode ─────────────────────────────────────────────────
    full_text: str = raw["choices"][0]["text"]  # type: ignore[index]
    return _strip_prompt_leak(full_text)
