#!/usr/bin/env python3
"""
Embedding Service for Fashion Search
Simplified version for Next.js integration
"""

import os
import sys
import json
import base64
import io
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

# Add the embedding directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our embedding functions
from image_search.query import search_image
from image_search.query_avg import search_topk, average_amount_sold
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, index_dir: str = None):
        """Initialize the embedding service"""
        if index_dir is None:
            index_dir = os.path.join(current_dir, "archive", "vector_index_cleaned")
        
        self.index_dir = index_dir
        self.model_name = "clip-ViT-B-32"
        self.device = "cpu"
        # Trend labels cache
        self._trend_labels_cache: Dict[str, Any] = {"labels": None, "embeddings": None, "model_name": self.model_name}
        
        # Verify index exists
        if not os.path.exists(os.path.join(index_dir, "image_index.faiss")):
            raise FileNotFoundError(f"Index not found at {index_dir}")

    def _load_trend_labels(self, labels: Optional[List[str]] = None, labels_path: Optional[str] = None) -> List[str]:
        if labels:
            return labels
        labels_path = labels_path or os.getenv("TRENDS_LABELS_PATH")
        if not labels_path:
            import glob
            pattern = os.path.join(os.getcwd(), "trends_labels_*.txt")
            files = sorted(glob.glob(pattern))
            if files:
                labels_path = files[-1]
        if labels_path and os.path.exists(labels_path):
            out: List[str] = []
            seen = set()
            with open(labels_path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = (ln or "").strip()
                    if s and s not in seen:
                        out.append(s)
                        seen.add(s)
            return out
        return [
            "black leather jacket",
            "wide leg jeans",
            "oversized blazer",
            "puffer jacket",
            "denim skirt",
        ]
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 data URI for web display"""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                # Resize to higher quality for better display
                img.thumbnail((600, 600), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=95)
                img_bytes = buffer.getvalue()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return ""
    
    def _add_image_urls(self, items: List[Dict]) -> List[Dict]:
        """Add image URLs and base64 data URIs to items for web display"""
        result = []
        for item in items:
            item_copy = item.copy()
            image_path = item.get("image_path")
            if image_path:
                # Extract image ID from the path for URL generation
                image_filename = os.path.basename(image_path)
                image_id = os.path.splitext(image_filename)[0]
                item_copy["image_url"] = f"/api/embedding/image/{image_id}"
                
                # Only add base64 data if image exists locally (for development)
                if os.path.exists(image_path):
                    item_copy["image_data_uri"] = self._encode_image_to_base64(image_path)
                else:
                    # For production/Vercel, we'll need to fetch images from external source
                    item_copy["image_data_uri"] = ""
            result.append(item_copy)
        return result
    
    def search_similar(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar items with deduplication"""
        try:
            # Search for more results to account for deduplication
            search_k = min(top_k * 3, 50)  # Search 3x more to find unique products
            results = search_image(
                query_image_path=image_path,
                index_dir=self.index_dir,
                top_k=search_k,
                model_name=self.model_name,
                device=self.device
            )
            
            # Deduplicate by product ID, keeping the best match for each product
            unique_products = {}
            for item in results:
                product_id = item.get('id', '')
                if product_id and product_id not in unique_products:
                    unique_products[product_id] = item
                elif product_id and product_id in unique_products:
                    # Keep the item with higher similarity score
                    if item.get('score', 0) > unique_products[product_id].get('score', 0):
                        unique_products[product_id] = item
            
            # Convert back to list and limit to requested top_k
            results = list(unique_products.values())[:top_k]
            
            # Add image URLs and base64 data
            results = self._add_image_urls(results)
            
            return {
                "success": True,
                "items": results,
                "total": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "total": 0
            }
    
    def search_with_average(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar items and calculate average amount sold with deduplication"""
        try:
            # Search for more results to account for deduplication
            search_k = min(top_k * 3, 50)  # Search 3x more to find unique products
            results = search_topk(
                query_image_path=image_path,
                index_dir=self.index_dir,
                top_k=search_k,
                model_name=self.model_name,
                device=self.device
            )
            
            # Deduplicate by product ID, keeping the best match for each product
            unique_products = {}
            for item in results:
                product_id = item.get('id', '')
                if product_id and product_id not in unique_products:
                    unique_products[product_id] = item
                elif product_id and product_id in unique_products:
                    # Keep the item with higher similarity score
                    if item.get('score', 0) > unique_products[product_id].get('score', 0):
                        unique_products[product_id] = item
            
            # Convert back to list and limit to requested top_k
            results = list(unique_products.values())[:top_k]
            
            # Calculate average amount sold
            avg_amount = average_amount_sold(results)
            
            # Add image URLs and base64 data
            results = self._add_image_urls(results)
            
            return {
                "success": True,
                "items": results,
                "average_amount_sold": avg_amount,
                "total": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "average_amount_sold": 0,
                "total": 0
            }
    
    def batch_search(self, image_paths: List[str], top_k: int = 5) -> Dict[str, Any]:
        """Search for similar items for multiple images"""
        try:
            all_results = []
            individual_averages = []
            
            for image_path in image_paths:
                result = self.search_with_average(image_path, top_k)
                if result["success"]:
                    all_results.append({
                        "image_path": image_path,
                        "items": result["items"],
                        "average_amount_sold": result["average_amount_sold"]
                    })
                    individual_averages.append(result["average_amount_sold"])
            
            # Calculate overall average
            overall_average = np.mean(individual_averages) if individual_averages else 0
            
            return {
                "success": True,
                "results": all_results,
                "overall_average_amount_sold": overall_average,
                "total_images": len(image_paths),
                "successful_searches": len(all_results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "overall_average_amount_sold": 0,
                "total_images": len(image_paths),
                "successful_searches": 0
            }

    def top_trend_matches(self, image_paths: List[str], top_k_items: int = 3, labels: Optional[List[str]] = None, labels_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            labels_list = self._load_trend_labels(labels, labels_path)
            if not labels_list:
                return {"success": False, "error": "No trend labels available", "top_items": []}

            cache = self._trend_labels_cache
            if (
                cache.get("labels") is None
                or cache.get("model_name") != self.model_name
                or cache.get("labels") != labels_list
            ):
                txt_model = SentenceTransformer(self.model_name, device=self.device)
                txt_embs = txt_model.encode(labels_list, convert_to_numpy=True, normalize_embeddings=True)
                cache.update({"labels": labels_list, "embeddings": txt_embs, "model_name": self.model_name})
            else:
                txt_embs = cache["embeddings"]

            img_model = SentenceTransformer(self.model_name, device=self.device)
            pil_images = []
            valid_paths: List[str] = []
            for p in image_paths:
                try:
                    with Image.open(p) as im:
                        pil_images.append(im.convert("RGB").copy())
                        valid_paths.append(p)
                except Exception:
                    pass
            if not pil_images:
                return {"success": False, "error": "No valid images", "top_items": []}

            img_embs = img_model.encode(pil_images, convert_to_numpy=True, normalize_embeddings=True)
            for im in pil_images:
                try:
                    im.close()
                except Exception:
                    pass

            sims = img_embs @ txt_embs.T
            best: List[Dict[str, Any]] = []
            for i, path in enumerate(valid_paths):
                j = int(np.argmax(sims[i]))
                score = float(sims[i][j])
                item = {"image_path": path, "best_trend": labels_list[j], "score": score}
                item = self._add_image_urls([item])[0]
                best.append(item)
            best.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return {"success": True, "top_items": best[:max(1, top_k_items)], "labels_count": len(labels_list), "model": self.model_name}
        except Exception as e:
            return {"success": False, "error": str(e), "top_items": []}

# Global service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

if __name__ == "__main__":
    # Test the service
    service = EmbeddingService()
    print("Embedding service initialized successfully!")
    print(f"Index directory: {service.index_dir}")