#!/usr/bin/env python3
"""
Lightweight Embedding Service for 512MB Render limit
Uses pre-computed embeddings instead of live model inference
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

class LightEmbeddingService:
    def __init__(self, index_dir: str = None):
        """Initialize the lightweight embedding service"""
        if index_dir is None:
            index_dir = os.path.join(current_dir, "archive", "vector_index_cleaned")
        
        self.index_dir = index_dir
        
        # Verify index exists
        if not os.path.exists(os.path.join(index_dir, "image_index.faiss")):
            raise FileNotFoundError(f"Index not found at {index_dir}")
    
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
        """Search for similar items using pre-computed embeddings"""
        try:
            # For now, return a mock response since we can't load the model
            # In production, you'd use a separate embedding service or pre-computed vectors
            return {
                "success": False,
                "error": "Lightweight service - model inference not available. Use full service for production.",
                "items": [],
                "total": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "items": [],
                "total": 0
            }
    
    def search_with_average(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar items and calculate average amount sold"""
        try:
            result = self.search_similar(image_path, top_k)
            if result["success"]:
                # Calculate average amount sold
                items = result["items"]
                amounts = [item.get("amount_sold", 0) for item in items if isinstance(item.get("amount_sold"), (int, float))]
                avg_amount = sum(amounts) / len(amounts) if amounts else 0
                
                return {
                    "success": True,
                    "items": items,
                    "average_amount_sold": avg_amount,
                    "total": len(items)
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "items": [],
                    "average_amount_sold": 0,
                    "total": 0
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

# Global service instance
_embedding_service = None

def get_embedding_service():
    """Get or create the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = LightEmbeddingService()
    return _embedding_service

if __name__ == "__main__":
    # Test the service
    service = LightEmbeddingService()
    print("Lightweight embedding service initialized successfully!")
    print(f"Index directory: {service.index_dir}")
