"""
HF Inference API Embedder
=========================
Drop-in replacement for SentenceTransformer.encode() that calls
Hugging Face's free Inference API instead of loading the model locally.

Used in LITE_MODE on Render to avoid ~300 MB of PyTorch RAM.
The .encode() method returns the same numpy array shape as
SentenceTransformer so FAISS code works unchanged.
"""

import os
import time
import requests
import numpy as np
from typing import List, Union


class HFInferenceEmbedder:
    """
    Mimics SentenceTransformer.encode() via HF Inference API.

    Usage:
        model = HFInferenceEmbedder('all-MiniLM-L6-v2')
        embeddings = model.encode(["hello world"])
        # → np.ndarray of shape (1, 384)
    """

    API_BASE = "https://api-inference.huggingface.co/models"
    MAX_RETRIES = 4
    RETRY_DELAY = 5  # seconds between retries on 503 (cold start)

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Normalise: allow both 'all-MiniLM-L6-v2' and
        # 'sentence-transformers/all-MiniLM-L6-v2'
        if '/' not in model_name:
            model_name = f"sentence-transformers/{model_name}"
        self.model_name = model_name
        self.api_url = f"{self.API_BASE}/{self.model_name}"
        self.token = os.environ.get('HF_TOKEN', '')
        if not self.token:
            print("[HFEmbedder] WARNING: HF_TOKEN not set — API calls may be rate-limited")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        print(f"[HFEmbedder] Using HF Inference API: {self.model_name}")

    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode sentences into embeddings via HF Inference API.

        Args:
            sentences: A string or list of strings to encode.
            **kwargs: Ignored (compatibility with SentenceTransformer.encode kwargs
                      like show_progress_bar, batch_size, etc.)

        Returns:
            np.ndarray of shape (n_sentences, embedding_dim)
        """
        # Handle single string input
        if isinstance(sentences, str):
            sentences = [sentences]

        # HF Inference API supports batching natively
        # but large batches can timeout — chunk into groups of 64
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call HF Inference API with retry logic for cold starts."""
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                r = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )

                if r.status_code == 200:
                    result = r.json()
                    # HF returns list of embeddings for sentence-transformers
                    if isinstance(result, list) and len(result) > 0:
                        # Could be [[float, ...], ...] or [float, ...]
                        if isinstance(result[0], list):
                            return result
                        else:
                            # Single sentence returned flat
                            return [result]

                elif r.status_code == 503:
                    # Model is loading (cold start)
                    wait_time = self.RETRY_DELAY * (attempt + 1)
                    print(f"[HFEmbedder] Model loading (503), retrying in {wait_time}s "
                          f"(attempt {attempt + 1}/{self.MAX_RETRIES})...")
                    time.sleep(wait_time)
                    continue

                elif r.status_code == 429:
                    # Rate limited
                    wait_time = 10 * (attempt + 1)
                    print(f"[HFEmbedder] Rate limited (429), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                else:
                    print(f"[HFEmbedder] API error {r.status_code}: {r.text[:200]}")
                    raise RuntimeError(f"HF API error {r.status_code}: {r.text[:200]}")

            except requests.exceptions.Timeout:
                print(f"[HFEmbedder] Timeout on attempt {attempt + 1}/{self.MAX_RETRIES}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    continue
                raise

        raise RuntimeError(f"HF API failed after {self.MAX_RETRIES} retries")
