import os
import urllib.request
from pathlib import Path

EMBEDDING_URL = "https://huggingface.co/datasets/guransh-goyal/redrob26-embeddings/resolve/main/bge_embeddings_completed.npz"
OUTPUT_PATH = "bge_embeddings_completed.npz"

def download_embeddings():
    if os.path.exists(OUTPUT_PATH):
        print(f"{OUTPUT_PATH} already exists.")
        return
    print(f"Downloading embeddings from {EMBEDDING_URL} ...")
    urllib.request.urlretrieve(EMBEDDING_URL, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    download_embeddings()
