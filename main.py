from fastapi import FastAPI
from pydantic import BaseModel
import base64, os, tempfile
from typing import List, Optional

# Copy your EmbeddingService into this repo as embedding_service.py (see below)
from embedding_service import get_embedding_service

app = FastAPI(title="Modo Embedding Service")

class SearchRequest(BaseModel):
  image_data: str
  top_k: int = 5

class BatchRequest(BaseModel):
  images_data: List[str]
  top_k: int = 5

def _write_temp_image(b64: str) -> str:
  if b64.startswith("data:"):
    b64 = b64.split(",", 1)[1]
  path = tempfile.mkstemp(suffix=".jpg")[1]
  with open(path, "wb") as f:
    f.write(base64.b64decode(b64))
  return path

@app.get("/health")
def health():
  try:
    svc = get_embedding_service()
    ok = os.path.exists(os.path.join(svc.index_dir, "image_index.faiss"))
    return {"status": "ok", "index_dir": svc.index_dir, "index_exists": ok}
  except Exception as e:
    return {"status": "error", "error": str(e)}

@app.post("/search")
def search(req: SearchRequest):
  svc = get_embedding_service()
  p = _write_temp_image(req.image_data)
  try:
    return svc.search_similar(p, req.top_k)
  finally:
    try: os.remove(p)
    except: pass

@app.post("/avg")
def avg(req: SearchRequest):
  svc = get_embedding_service()
  p = _write_temp_image(req.image_data)
  try:
    return svc.search_with_average(p, req.top_k)
  finally:
    try: os.remove(p)
    except: pass

@app.post("/batch")
def batch(req: BatchRequest):
  svc = get_embedding_service()
  paths = []
  try:
    for b in req.images_data:
      paths.append(_write_temp_image(b))
    return svc.batch_search(paths, req.top_k)
  finally:
    for p in paths:
      try: os.remove(p)
      except: pass