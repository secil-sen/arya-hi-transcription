# -*- coding: utf-8 -*-
"""
Embedding Extraction for Transcript Segments

This module extracts speaker embeddings from audio segments using pyannote.audio
embedding models. These embeddings can be used for:
- Speaker verification and clustering
- Voice similarity analysis
- Speaker re-identification across sessions
- Enhanced speaker diarization quality

The embeddings are extracted using the segment's audio timestamps and stored
as numpy arrays in the segment metadata.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import torch

logger = logging.getLogger(__name__)


def extract_embeddings_from_segments(
    segments: List[Dict[str, Any]],
    audio_path: str,
    inference_model=None
) -> List[Dict[str, Any]]:
    """
    Extract speaker embeddings from audio segments.

    Args:
        segments: List of transcript segments with start/end timestamps
        audio_path: Path to the audio file (WAV format)
        inference_model: Optional pre-loaded inference model (from model_registry)

    Returns:
        List of segments with added 'embedding' field containing numpy arrays

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ImportError: If pyannote.audio is not installed
    """

    if not segments:
        logger.warning("No segments provided for embedding extraction")
        return segments

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Import pyannote components
    try:
        from pyannote.audio import Inference, Model
        from pyannote.core import Segment as PyannoteSegment
    except ImportError as e:
        logger.error(f"pyannote.audio not available: {e}")
        logger.error("Install with: pip install 'pyannote.audio>=3.1.1'")
        # Return segments without embeddings
        for seg in segments:
            seg["embedding"] = None
            seg["embedding_error"] = str(e)
        return segments

    # Load inference model if not provided
    if inference_model is None:
        logger.info("Loading embedding model from model_registry...")
        from app.core.model_registry import models
        inference_model = models.inference

        if inference_model is None:
            logger.warning("Inference model not available in registry, loading manually...")
            try:
                from app.core.config import EMBEDDING_MODEL_NAME
                hf_token = os.getenv("HUGGINGFACE_TOKEN")

                if not hf_token:
                    logger.error("HUGGINGFACE_TOKEN not set")
                    for seg in segments:
                        seg["embedding"] = None
                        seg["embedding_error"] = "Missing HuggingFace token"
                    return segments

                # Try Inference class first (pyannote 3.x)
                try:
                    inference_model = Inference(
                        EMBEDDING_MODEL_NAME,
                        use_auth_token=hf_token
                    )
                    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    for seg in segments:
                        seg["embedding"] = None
                        seg["embedding_error"] = f"Model loading failed: {e}"
                    return segments

            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                for seg in segments:
                    seg["embedding"] = None
                    seg["embedding_error"] = str(e)
                return segments

    # Extract embeddings for each segment
    logger.info(f"Extracting embeddings for {len(segments)} segments...")
    start_time = __import__('time').time()

    enriched_segments = []
    successful_extractions = 0
    failed_extractions = 0

    for i, segment in enumerate(segments):
        try:
            start_sec = float(segment.get("start", 0.0))
            end_sec = float(segment.get("end", 0.0))

            # Validate timestamps
            if start_sec >= end_sec:
                logger.warning(f"Segment {i}: Invalid timestamps (start={start_sec}, end={end_sec})")
                segment["embedding"] = None
                segment["embedding_error"] = "Invalid timestamps"
                enriched_segments.append(segment)
                failed_extractions += 1
                continue

            # Create pyannote Segment
            pyannote_segment = PyannoteSegment(start_sec, end_sec)

            # Extract embedding
            # The inference model returns different types depending on pyannote version
            raw_embedding = inference_model({"audio": audio_path, "segment": pyannote_segment})

            # Handle different return types from pyannote
            embedding = None

            # Type 1: SlidingWindowFeature (pyannote 3.x)
            if hasattr(raw_embedding, 'data'):
                # SlidingWindowFeature has a .data attribute that's a numpy array
                embedding = raw_embedding.data
                # Average across time dimension if needed (shape: [time, features])
                if len(embedding.shape) > 1:
                    embedding = np.mean(embedding, axis=0)  # Average pooling
            # Type 2: Torch tensor
            elif isinstance(raw_embedding, torch.Tensor):
                embedding = raw_embedding.cpu().detach().numpy()
            # Type 3: Numpy array
            elif isinstance(raw_embedding, np.ndarray):
                embedding = raw_embedding
            else:
                # Unknown type, try to convert
                try:
                    embedding = np.array(raw_embedding)
                except Exception as conv_error:
                    logger.error(f"Segment {i}: Unknown embedding type {type(raw_embedding)}: {conv_error}")
                    segment["embedding"] = None
                    segment["embedding_error"] = f"Unknown embedding type: {type(raw_embedding)}"
                    enriched_segments.append(segment)
                    failed_extractions += 1
                    continue

            # Ensure it's 1D array
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()

            # Store embedding in segment
            segment["embedding"] = embedding.tolist()  # Convert to list for JSON serialization
            segment["embedding_shape"] = list(embedding.shape)
            segment["embedding_dim"] = len(embedding)

            successful_extractions += 1
            enriched_segments.append(segment)

        except Exception as e:
            logger.warning(f"Segment {i}: Embedding extraction failed - {e}")
            segment["embedding"] = None
            segment["embedding_error"] = str(e)
            enriched_segments.append(segment)
            failed_extractions += 1

    elapsed_time = __import__('time').time() - start_time

    # Logging summary
    logger.info(f"✅ Embedding extraction completed in {elapsed_time:.2f}s")
    logger.info(f"   - Successful: {successful_extractions}/{len(segments)} segments")
    logger.info(f"   - Failed: {failed_extractions}/{len(segments)} segments")

    if successful_extractions > 0:
        sample_dim = enriched_segments[0].get("embedding_dim", "unknown")
        logger.info(f"   - Embedding dimension: {sample_dim}")

    return enriched_segments


async def extract_embeddings_async(
    segments: List[Dict[str, Any]],
    audio_path: str,
    inference_model=None
) -> List[Dict[str, Any]]:
    """
    Async wrapper for embedding extraction.

    This is useful when the extraction is called from an async context
    (like the main pipeline).

    Args:
        segments: List of transcript segments
        audio_path: Path to the audio file
        inference_model: Optional pre-loaded inference model

    Returns:
        List of segments with embeddings
    """
    import asyncio

    # Run the synchronous function in a thread pool
    return await asyncio.to_thread(
        extract_embeddings_from_segments,
        segments,
        audio_path,
        inference_model
    )


def compute_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity score (0-1, higher means more similar)
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    # Convert lists to numpy arrays if needed
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)

    # Compute cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def cluster_speakers_by_embeddings(
    segments: List[Dict[str, Any]],
    similarity_threshold: float = 0.75
) -> Dict[str, List[int]]:
    """
    Cluster segments by speaker using embedding similarity.

    This is a simple greedy clustering algorithm that groups segments
    with similar embeddings together.

    Args:
        segments: List of segments with embeddings
        similarity_threshold: Minimum similarity to consider same speaker

    Returns:
        Dictionary mapping cluster_id to list of segment indices
    """

    if not segments:
        return {}

    # Filter segments with valid embeddings
    valid_segments = [
        (i, seg) for i, seg in enumerate(segments)
        if seg.get("embedding") is not None
    ]

    if not valid_segments:
        logger.warning("No segments with valid embeddings for clustering")
        return {}

    clusters = {}
    cluster_id = 0

    for idx, segment in valid_segments:
        embedding = np.array(segment["embedding"])

        # Find best matching cluster
        best_cluster = None
        best_similarity = 0.0

        for cid, cluster_indices in clusters.items():
            # Compare with first segment in cluster (representative)
            rep_idx = cluster_indices[0]
            rep_embedding = np.array(segments[rep_idx]["embedding"])

            similarity = compute_embedding_similarity(embedding, rep_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cid

        # Assign to cluster or create new one
        if best_cluster is not None and best_similarity >= similarity_threshold:
            clusters[best_cluster].append(idx)
        else:
            clusters[f"cluster_{cluster_id}"] = [idx]
            cluster_id += 1

    logger.info(f"Clustered {len(valid_segments)} segments into {len(clusters)} speaker clusters")
    return clusters