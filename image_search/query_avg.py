import os
import json
from typing import List

import faiss  # type: ignore
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .query import load_index as load_index_from_query

# Constrain threads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def load_index(index_dir: str):
    # Backwards-compatible shim that delegates to query.load_index (now supports URLs).
    return load_index_from_query(index_dir)  # type: ignore


def search_topk(query_image_path: str, index_dir: str, top_k: int, model_name: str = "clip-ViT-B-32", device: str = "cpu") -> List[dict]:
    if not os.path.isfile(query_image_path):
        raise FileNotFoundError(query_image_path)
    
    # Use cached index and model from query.py
    from .query import _index_cache, _model_cache
    cache_key = f"{index_dir}_{model_name}_{device}"
    if cache_key not in _index_cache:
        _index_cache[cache_key] = load_index(index_dir)
    index, meta = _index_cache[cache_key]
    
    if cache_key not in _model_cache:
        # Fix meta tensor device issue
        import torch
        # Ensure we're using CPU properly
        if device == "cpu":
            torch.set_default_device("cpu")
            torch.set_default_tensor_type(torch.FloatTensor)
        try:
            _model_cache[cache_key] = SentenceTransformer(model_name, device=device)
        except Exception as e:
            # Fallback: try loading without device specification
            print(f"Model loading failed with device={device}, trying without device: {e}")
            _model_cache[cache_key] = SentenceTransformer(model_name)
    model = _model_cache[cache_key]

    img = Image.open(query_image_path).convert("RGB")
    q = model.encode([img], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, idxs = index.search(q, top_k)
    results: List[dict] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        item = meta[idx]
        item = {**item, "score": float(score)}
        results.append(item)
    return results


def average_amount_sold(items: List[dict]) -> float:
    vals: List[float] = []
    for it in items:
        if "amount_sold" in it and isinstance(it["amount_sold"], (int, float)):
            vals.append(float(it["amount_sold"]))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Return average amount_sold of top-k similar items for an image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--index_dir", default=os.getenv("INDEX_SOURCE", os.getenv("INDEX_DIR", "vector_index")))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only_avg", action="store_true", help="Print only the numeric average amount_sold")
    args = parser.parse_args()

    hits = search_topk(args.image, args.index_dir, top_k=args.top_k, model_name=args.model, device=args.device)
    avg = average_amount_sold(hits)
    if args.only_avg:
        # Print just the float for easy piping
        print(f"{avg}")
    else:
        print(json.dumps({
            "query_image": args.image,
            "top_k": args.top_k,
            "average_amount_sold": avg,
            "items": hits,
        }, ensure_ascii=False))
