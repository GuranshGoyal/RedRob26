"""Unified embedding retrieval/generation.

This module hides whether the BGE embedding cache is downloaded from a remote URL or generated
locally. Consumers (rank.py, the notebook) simply call ensure_cache() and receive the path to a valid
cache file. The cache is keyed by candidate_id + text hash so that stale rows are re-encoded
automatically when the data or the relevance-first text logic changes.
"""

import gzip
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

EMBEDDING_URL = (
    "https://huggingface.co/datasets/guransh-goyal/redrob26-embeddings/"
    "resolve/main/bge_embeddings_completed.npz"
)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MAX_TOKENS = 512
EMBED_MAX_WORDS = 400
CHECKPOINT_EVERY = 5000

# ---------------------------------------------------------------------------
# The text-building logic below must match the logic used in rank.py so that
# the text hash stored in the cache is consistent with the text the pipeline
# uses when it looks up candidate vectors.
# ---------------------------------------------------------------------------
IR_CORE_KEYWORDS = [
    "retrieval", "ranking", "recommendation", "relevance", "search", "embedding", "embeddings",
    "semantic search", "vector", "bm25", "faiss", "learning to rank", "recommender", "matching",
]

EVAL_KEYWORDS = [
    "ndcg", "mrr", "map", "evaluation", "offline", "online", "benchmark",
    "a/b", "ab test", "ab testing", "precision", "recall",
]

LLM_KEYWORDS = [
    "prompt engineering", "fine-tun", "lora", "qlora", "peft", "llm", "rag",
    "large language model", "transformer", "deployment",
]

AIML_SKILLS = {
    "information retrieval", "information retrieval systems", "semantic search", "vector search",
    "recommendation systems", "ranking systems", "learning to rank", "bm25", "faiss", "pinecone",
    "weaviate", "milvus", "qdrant", "pgvector", "elasticsearch", "opensearch", "haystack",
    "search backend", "search infrastructure", "search & discovery", "indexing algorithms",
    "content matching", "vector representations", "text encoders",
    "embeddings", "sentence transformers", "hugging face transformers", "llms", "llm", "rag",
    "langchain", "llamaindex", "prompt engineering", "fine-tuning llms", "lora", "qlora", "peft",
    "nlp", "natural language processing", "model adaptation",
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn", "data science",
    "feature engineering", "statistical modeling", "reinforcement learning", "time series",
    "forecasting", "model adaptation",
    "mlops", "mlflow", "kubeflow", "bentoml", "weights & biases", "open-source ml libraries",
}

_EMBED_PRIORITY = set(IR_CORE_KEYWORDS) | set(EVAL_KEYWORDS) | set(LLM_KEYWORDS) | set(AIML_SKILLS)


def _open_any(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "r", encoding="utf-8")


def _lower(x):
    return str(x).lower() if x is not None else ""


def _dedupe_cap(text, max_words=EMBED_MAX_WORDS):
    seen, out = set(), []
    for w in text.split():
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_words:
            break
    return " ".join(out)


def _build_embed_text(sk, ct, sm, ta, hl, cr):
    skills = sk.split()
    rel = [s for s in skills if any(p in s for p in _EMBED_PRIORITY)]
    rel_set = set(rel)
    oth = [s for s in skills if s not in rel_set]
    parts = [" ".join(rel + oth), ct, sm, ta.replace("|", " "), hl, cr]
    return _dedupe_cap(" ".join(p for p in parts if p))


def _txt_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _save_cache(cache: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _ids = np.array(list(cache.keys()))
    _hashes = np.array([cache[c][0] for c in _ids])
    _vecs = np.vstack([cache[c][1] for c in _ids]).astype(np.float32)
    np.savez(path, ids=_ids, hashes=_hashes, vecs=_vecs)


def _download(url: str, cache_path: str, timeout: int = 300) -> bool:
    """Try to download the pre-computed cache. Return True on success."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        print(f"Downloading embeddings from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp_path, "wb") as out_file:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
        os.replace(tmp_path, cache_path)
        print(f"Saved to {cache_path}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        print(f"Download failed: {e}")
        if os.path.exists(cache_path + ".tmp"):
            try:
                os.remove(cache_path + ".tmp")
            except OSError:
                pass
        return False


def _generate(cache_path: str, data_path: str) -> bool:
    """Generate the embedding cache locally from the candidate pool. Return True on success."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Local generation requires sentence-transformers: {e}")
        return False

    if not os.path.exists(data_path):
        print(f"Cannot generate embeddings: data file not found: {data_path}")
        return False

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache = {}
    if os.path.exists(cache_path):
        _z = np.load(cache_path, allow_pickle=True)
        cache = {str(cid): (str(h), v) for cid, h, v in zip(_z["ids"], _z["hashes"], _z["vecs"])}
        print(f"Loaded existing partial cache: {len(cache):,} vectors.")

    need_ids, need_text = [], []
    t0 = time.time()
    with _open_any(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            prof = c.get("profile", {}) or {}
            skills = c.get("skills", []) or []
            career = c.get("career_history", []) or []
            cid = str(c.get("candidate_id"))

            headline = _lower(prof.get("headline"))
            summary = _lower(prof.get("summary"))
            cur_title = _lower(prof.get("current_title"))
            skill_names = [_lower(s.get("name")) for s in skills]
            skills_text = " ".join(skill_names)
            titles_all = " | ".join([cur_title] + [_lower(e.get("title")) for e in career])
            career_text = " ".join(_lower(e.get("description")) for e in career)
            embed_text = _build_embed_text(skills_text, cur_title, summary, titles_all, headline, career_text)
            h = _txt_hash(embed_text)
            if cid not in cache or cache[cid][0] != h:
                need_ids.append(cid)
                need_text.append(embed_text)

    print(f"Parsed full pool in {time.time() - t0:.1f}s; {len(need_ids):,} candidates need fresh embeddings.")

    if not need_ids:
        print(f"All {len(cache):,} candidates already cached.")
        return True

    print(f"Loading embedding model {EMBED_MODEL} ...")
    st_model = SentenceTransformer(EMBED_MODEL)
    st_model.max_seq_length = EMBED_MAX_TOKENS

    ts = time.time()
    for start in range(0, len(need_text), CHECKPOINT_EVERY):
        batch_ids = need_ids[start:start + CHECKPOINT_EVERY]
        batch_text = need_text[start:start + CHECKPOINT_EVERY]
        batch_vecs = st_model.encode(
            batch_text, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
        for cid, t, v in zip(batch_ids, batch_text, batch_vecs):
            cache[cid] = (_txt_hash(t), np.asarray(v, dtype=np.float32))
        _save_cache(cache, cache_path)
        print(
            f"Checkpoint saved: {len(cache):,} total vectors "
            f"({start + len(batch_ids):,} / {len(need_ids):,} encoded)."
        )
    print(f"Embedded {len(need_ids):,} candidates in {time.time() - ts:.1f}s.")
    return True


def ensure_cache(
    cache_path: str = "cache/bge_embeddings_completed.npz",
    data_path: str = "data/candidates.jsonl",
    url: str = EMBEDDING_URL,
    force: bool = False,
) -> str:
    """Return the path to a valid embedding cache, downloading or generating it if needed.

    Order of operations:
      1. If cache exists and force is False, return it immediately.
      2. Try to download the cache from the provided URL.
      3. If download fails, try to generate the cache locally from data_path.
      4. If both fail, raise RuntimeError.
    """
    if os.path.exists(cache_path) and not force:
        print(f"Embedding cache already exists: {cache_path}")
        return cache_path

    if _download(url, cache_path):
        return cache_path

    print("Falling back to local generation ...")
    if _generate(cache_path, data_path):
        return cache_path

    raise RuntimeError(
        f"Could not obtain embedding cache at {cache_path}. "
        "Download failed and local generation also failed. "
        "Please check your internet connection or ensure the data file is available."
    )


if __name__ == "__main__":
    ensure_cache()
