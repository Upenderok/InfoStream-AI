"""
Quick warm-up to ensure FAISS, encoder & LLM load without UI latency.
"""

from src.retriever import Retriever
from src.generator import generate

print("⏳ Loading retriever …")
retr = Retriever()
print("⏳ Priming LLM …")
_ = next(generate("You are a test.", "Context", "Hello?", stream=True))
print("✅ All resources loaded & primed.")
print("You can now run `streamlit run app.py` to start the chat UI.")