# download_phi3.py
from huggingface_hub import hf_hub_download

repo_id  = "microsoft/Phi-3-mini-4k-instruct-gguf"
filename = "Phi-3-mini-4k-instruct-q4.gguf"

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    cache_dir="models",      # saves to ./models/
    library_name="llama_cpp_python"
)
print(f"✅ Downloaded model to {local_path}")
