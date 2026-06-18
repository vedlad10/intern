"""
HF Inference API Embedder
=========================
Drop-in replacement for SentenceTransformer.encode() that calls
Hugging Face's Inference API via the official huggingface_hub library.

Used in LITE_MODE on Render to avoid ~300 MB of PyTorch RAM.
The .encode() method returns the same numpy array shape as
SentenceTransformer so FAISS code works unchanged.
"""

import os
import time
import numpy as np
from typing import List, Union


class HFInferenceEmbedder:
    """
    Mimics SentenceTransformer.encode() via HF Inference API.

    Uses the official huggingface_hub InferenceClient which handles
    endpoint routing automatically (no hardcoded URLs to break).

    Usage:
        model = HFInferenceEmbedder('all-MiniLM-L6-v2')
        embeddings = model.encode(["hello world"])
        # → np.ndarray of shape (1, 384)
    """

    MAX_RETRIES = 4
    RETRY_DELAY = 5  # seconds between retries on cold start

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        from huggingface_hub import InferenceClient

        # Normalise: allow both 'all-MiniLM-L6-v2' and
        # 'sentence-transformers/all-MiniLM-L6-v2'
        if '/' not in model_name:
            model_name = f"sentence-transformers/{model_name}"
        self.model_name = model_name

        token = os.environ.get('HF_TOKEN', '')
        if not token:
            print("[HFEmbedder] WARNING: HF_TOKEN not set — API calls may be rate-limited")

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=token
        )
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

        # Process in batches to avoid timeouts on large inputs
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call HF Inference API with retry logic for cold starts."""
        for attempt in range(self.MAX_RETRIES):
            try:
                # feature_extraction returns embeddings for each input text
                result = self.client.feature_extraction(
                    texts,
                    model=self.model_name
                )

                # Result can be a numpy array or nested list
                if hasattr(result, 'tolist'):
                    result = result.tolist()

                # Handle different response shapes:
                # Single text  → [float, ...] or [[float, ...]]
                # Multi text   → [[float, ...], [float, ...]]
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], (int, float)):
                        # Single flat embedding — wrap it
                        return [result]
                    elif isinstance(result[0], list):
                        # Check if it's [[[float]]] (extra nesting from some models)
                        if isinstance(result[0][0], list):
                            # Token-level embeddings — mean pool to sentence level
                            return [
                                np.mean(token_embeds, axis=0).tolist()
                                for token_embeds in result
                            ]
                        return result

                raise RuntimeError(f"Unexpected API response shape: {type(result)}")

            except Exception as e:
                error_str = str(e).lower()

                # Retryable errors: model loading, rate limits, timeouts
                if any(kw in error_str for kw in ['503', 'loading', '429', 'rate', 'timeout', 'timed out']):
                    wait_time = self.RETRY_DELAY * (attempt + 1)
                    print(f"[HFEmbedder] Retryable error: {e}")
                    print(f"[HFEmbedder] Retrying in {wait_time}s "
                          f"(attempt {attempt + 1}/{self.MAX_RETRIES})...")
                    time.sleep(wait_time)
                    continue

                # Non-retryable error — raise immediately
                print(f"[HFEmbedder] Fatal API error: {e}")
                raise

        raise RuntimeError(f"HF API failed after {self.MAX_RETRIES} retries")
