"""
ONNX Embedder — Lightweight local embeddings
=============================================
Drop-in replacement for SentenceTransformer.encode() using ONNX Runtime.

Runs the same all-MiniLM-L6-v2 model but with ONNX Runtime (~100 MB RAM)
instead of PyTorch (~400 MB RAM). Produces identical embeddings so
pre-built FAISS indices remain valid.

No external API calls — fully self-contained.
"""

import os
import numpy as np
from typing import List, Union


class HFInferenceEmbedder:
    """
    Mimics SentenceTransformer.encode() using ONNX Runtime locally.

    Usage:
        model = HFInferenceEmbedder('all-MiniLM-L6-v2')
        embeddings = model.encode(["hello world"])
        # → np.ndarray of shape (1, 384)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # Normalise model name
        if '/' not in model_name:
            model_name = f"sentence-transformers/{model_name}"
        self.model_name = model_name

        # Model files directory (downloaded during build)
        model_dir = os.environ.get('ONNX_MODEL_DIR', 'onnx_model')

        onnx_path = os.path.join(model_dir, 'model.onnx')
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                f"Run 'python download_model.py' first."
            )

        print(f"[ONNXEmbedder] Loading ONNX model from {model_dir}...")

        # Load ONNX model with optimized settings for low-memory
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        # Load fast tokenizer (Rust-based, very lightweight)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)

        # Get model input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        print(f"[ONNXEmbedder] Ready — {self.model_name} (ONNX)")

    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode sentences into embeddings using ONNX Runtime.

        Args:
            sentences: A string or list of strings to encode.
            **kwargs: Ignored (compatibility with SentenceTransformer.encode kwargs).

        Returns:
            np.ndarray of shape (n_sentences, embedding_dim)
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        # Process in batches to manage memory
        batch_size = kwargs.get('batch_size', 64)
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts using ONNX Runtime."""
        # Tokenize
        encodings = self.tokenizer.encode_batch(texts)

        # Build numpy arrays for model input
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Prepare feeds — only include inputs the model expects
        feeds = {}
        if 'input_ids' in self.input_names:
            feeds['input_ids'] = input_ids
        if 'attention_mask' in self.input_names:
            feeds['attention_mask'] = attention_mask
        if 'token_type_ids' in self.input_names:
            feeds['token_type_ids'] = token_type_ids

        # Run inference
        outputs = self.session.run(self.output_names, feeds)

        # outputs[0] is typically token_embeddings: (batch, seq_len, hidden_dim)
        # We need to mean-pool over the token dimension (like SentenceTransformers does)
        token_embeddings = outputs[0]

        # Mean pooling with attention mask
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        sentence_embeddings = sum_embeddings / sum_mask

        # Normalize (SentenceTransformers normalizes by default for this model)
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        sentence_embeddings = sentence_embeddings / norms

        return sentence_embeddings.astype(np.float32)
