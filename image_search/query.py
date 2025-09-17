import os
# Constrain thread counts to avoid segfaults on macOS/Python 3.13
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import hashlib
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import requests


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _cache_dir_for(url_base: str) -> str:
    root = os.path.join(os.path.expanduser("~"), ".cache", "embedding_index")
    os.makedirs(root, exist_ok=True)
    h = hashlib.sha256(url_base.encode("utf-8")).hexdigest()[:16]
    d = os.path.join(root, h)
    os.makedirs(d, exist_ok=True)
    return d


def _download_file(url: str, out_path: str) -> None:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(resp.content)


def _ensure_local_index(source: str) -> Tuple[str, str]:
    """Return local paths (index_path, meta_path) for either a local dir or an HTTP base URL.

    If source is a directory, assumes files named image_index.faiss and image_meta.json exist in it.
    If source is a URL, expects those filenames at `${source}/image_index.faiss` and `${source}/image_meta.json`.
    Downloads to ~/.cache/embedding_index/<hash>/ and returns those local paths.
    """
    if not _is_url(source):
        index_path = os.path.join(source, "image_index.faiss")
        meta_path = os.path.join(source, "image_meta.json")
        return index_path, meta_path

    base = source.rstrip("/")
    cache_dir = _cache_dir_for(base)
    index_path = os.path.join(cache_dir, "image_index.faiss")
    meta_path = os.path.join(cache_dir, "image_meta.json")

    # Download if missing
    if not os.path.exists(index_path):
        _download_file(f"{base}/image_index.faiss", index_path)
    if not os.path.exists(meta_path):
        _download_file(f"{base}/image_meta.json", meta_path)
    return index_path, meta_path


def load_index(index_source: str):
    index_path, meta_path = _ensure_local_index(index_source)
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("Index or metadata not found. Build or provide the index first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return index, meta


# Global model cache
_model_cache = {}
_index_cache = {}

def search_image(query_image_path: str, index_dir: str, top_k: int = 5, model_name: str = "clip-ViT-B-32", device: str = "cpu") -> List[dict]:
    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    
    # Cache index and model
    cache_key = f"{index_dir}_{model_name}_{device}"
    if cache_key not in _index_cache:
        _index_cache[cache_key] = load_index(index_dir)
    index, meta = _index_cache[cache_key]
    
    if cache_key not in _model_cache:
        _model_cache[cache_key] = SentenceTransformer(model_name, device=device)
    model = _model_cache[cache_key]

    img = Image.open(query_image_path).convert("RGB")
    q = model.encode([img], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, idxs = index.search(q, top_k)
    results: List[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        item = meta[idx]
        results.append({
            "score": float(score),
            **item,
        })
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--index_dir", default="vector_index")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    hits = search_image(args.image, args.index_dir, top_k=args.top_k, model_name=args.model, device=args.device)
    for h in hits:
        print(json.dumps(h, ensure_ascii=False))
