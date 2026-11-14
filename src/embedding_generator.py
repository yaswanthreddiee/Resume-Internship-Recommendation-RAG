"""
Module 3: Embedding Generation Pipeline
Uses sentence-transformers/all-MiniLM-L6-v2 to generate dense semantic embeddings
for candidate resumes and internship postings.
Implements model caching (load once), batching, and cosine similarity calculation.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from src.resume_preprocessing import preprocess_resume
from src.internship_processing import (
    load_and_preprocess_internships,
    prepare_internship_embedding_text
)

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


class EmbeddingManager:
    """
    Singleton Manager ensuring the SentenceTransformer model is loaded only once
    into memory across the entire application lifecycle.
    """
    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: str = DEFAULT_MODEL_NAME

    def __new__(cls, model_name: str = DEFAULT_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._model_name = model_name
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        logger.info(f"Loading embedding model: {self._model_name}...")
        start_t = time.perf_counter()
        # Initialize SentenceTransformer (runs on CPU or MPS if available)
        self._model = SentenceTransformer(self._model_name)
        elapsed = time.perf_counter() - start_t
        logger.info(f"Model {self._model_name} loaded successfully in {elapsed:.2f}s.")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_text(self, text: str, normalize: bool = True) -> List[float]:
        """Encodes a single string into a normalized embedding vector."""
        emb = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
        return emb.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> List[List[float]]:
        """Encodes a batch of strings with efficient batch processing."""
        if not texts:
            return []
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embs.tolist()


def get_embedding_manager(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingManager:
    """Retrieves the global singleton EmbeddingManager instance."""
    return EmbeddingManager(model_name)


def generate_candidate_embedding(candidate_data: Union[Dict[str, Any], str]) -> List[float]:
    """
    Generates a dense semantic embedding for candidate resume information.

    Args:
        candidate_data: Raw or preprocessed candidate dictionary or JSON string.

    Returns:
        List[float]: 384-dimensional normalized embedding vector.
    """
    manager = get_embedding_manager()

    if isinstance(candidate_data, dict) and "embedding_text" in candidate_data and candidate_data["embedding_text"]:
        text = candidate_data["embedding_text"]
    else:
        _, text = preprocess_resume(candidate_data)

    return manager.encode_text(text, normalize=True)


def generate_internship_embeddings(
    internships: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 32
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """
    Generates embeddings for all internship records.
    Loads from data/internships.json if no list is provided.

    Returns:
        Tuple of (validated_internship_records, embeddings_list).
    """
    manager = get_embedding_manager()

    if internships is None:
        internships = load_and_preprocess_internships("data/internships.json")

    texts = []
    for rec in internships:
        if "embedding_text" not in rec or not rec["embedding_text"]:
            rec["embedding_text"] = prepare_internship_embedding_text(rec)
        texts.append(rec["embedding_text"])

    logger.info(f"Generating embeddings for {len(texts)} internship records (batch_size={batch_size})...")
    start_t = time.perf_counter()
    embeddings = manager.encode_batch(texts, batch_size=batch_size, normalize=True)
    elapsed = time.perf_counter() - start_t
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.3f}s ({elapsed/max(1, len(texts))*1000:.1f}ms/item).")

    return internships, embeddings


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Computes exact cosine similarity between two float vectors."""
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
