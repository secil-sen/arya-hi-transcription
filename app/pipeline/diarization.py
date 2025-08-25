"""
Diarization pipeline with overlapped chunking and global speaker ID assignment
using speaker embeddings and duration-weighted centroid updates.

This version adds an optional **KMeans-forced reclustering** step to enforce
the final number of speakers (e.g., attendee_num=2) even if the incremental
assignment produced more clusters due to noise.

- chunk_paths can be a directory path OR a precomputed list of file paths.
"""
import os
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from pyannote.core import Segment
from app.core.model_registry import models
from app.pipeline.config import CHUNK_LENGTH, CHUNK_OVERLAP
import torchaudio
from sklearn.cluster import KMeans
from collections import defaultdict

MIN_EMB_DUR = 0.5   # seconds - reduced from 1.2 to be more permissive
EXPAND_MARGIN = 0.6 # seconds

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def merge_segments(segments: List["Segment"], gap_tolerance: float = 0.3) -> List["Segment"]:
    """Merge adjacent segments of the same local speaker with small gaps tolerated."""
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s.start, s.end))
    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start - last.end <= gap_tolerance:
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged

def duration_weighted_mean(vectors: List[np.ndarray], durations: List[float]) -> np.ndarray:
    w = np.asarray(durations, dtype=np.float32)
    wsum = float(w.sum()) + 1e-12
    mat = np.vstack(vectors).astype(np.float32)
    return (mat.T @ (w / wsum)).T

def _audio_duration_seconds(path: str) -> Optional[float]:
    """Return audio duration in seconds using torchaudio if available."""
    if torchaudio is None:
        return None
    try:
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return None
    return None

def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norm

def compute_speaker_durations(segments: List[Dict]) -> Dict[str, float]:
    """Aggregate total duration per speaker id from segment list."""
    dur = defaultdict(float)
    for seg in segments:
        dur[seg["speaker"]] += float(seg["end"] - seg["start"])
    return dict(dur)

def _ensure_min_duration(seg: Segment, file_duration: Optional[float]) -> Segment:
    """If segment is shorter than MIN_EMB_DUR, expand it symmetrically within file bounds."""
    if seg.duration >= MIN_EMB_DUR or file_duration is None:
        return seg if seg.duration >= MIN_EMB_DUR else seg  # If duration unknown, keep as is (will be skipped later)
    need = max(MIN_EMB_DUR - seg.duration, 0.0)
    extra_each_side = max(EXPAND_MARGIN, need / 2.0)
    new_start = max(0.0, seg.start - extra_each_side)
    new_end   = min(file_duration, seg.end + extra_each_side)
    if (new_end - new_start) < MIN_EMB_DUR:
        center = (seg.start + seg.end) / 2.0
        half = MIN_EMB_DUR / 2.0
        new_start = max(0.0, center - half)
        new_end   = min(file_duration, center + half)
    return Segment(new_start, new_end)

# KMeans-forced reclustering
def force_num_speakers_kmeans(
    segments: List[Dict],
    global_speakers: Dict[str, np.ndarray],
    attendee_num: int
) -> Tuple[List[Dict], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Recluster existing global speaker centroids into exactly `attendee_num` clusters
    using KMeans, with sample weights proportional to each speaker's total speaking time.
    Returns:
      new_segments: segments with speakers remapped to spk_0..spk_{attendee_num-1}
      new_centroids: dict of enforced centroids (L2-normalized space)
      mapping: old_id -> new_id
    """

    if len(global_speakers) <= attendee_num:
        # Already <= target, no need to force
        return segments, global_speakers, {k: k for k in global_speakers.keys()}

    g_ids = list(global_speakers.keys())
    X = np.vstack([np.asarray(global_speakers[g]) for g in g_ids]).astype(np.float32)
    X = _l2_normalize(X)  # cosine-consistent space

    spk_durations = compute_speaker_durations(segments)
    weights = np.array([spk_durations.get(g, 1.0) for g in g_ids], dtype=np.float32)

    try:
        km = KMeans(n_clusters=attendee_num, n_init='auto', random_state=42)
    except TypeError:
        km = KMeans(n_clusters=attendee_num, n_init=10, random_state=42)

    try:
        km.fit(X, sample_weight=weights)
    except TypeError:
        km.fit(X)

    labels = km.labels_  # len = len(g_ids)

    new_id_map = {gid: f"spk_{lab}" for gid, lab in zip(g_ids, labels)}

    new_segments = []
    for seg in segments:
        old = seg["speaker"]
        seg2 = dict(seg)
        seg2["speaker"] = new_id_map.get(old, old)
        new_segments.append(seg2)

    new_centroids = {}
    for k in range(attendee_num):
        new_centroids[f"spk_{k}"] = km.cluster_centers_[k].astype(np.float32)

    return new_segments, new_centroids, new_id_map

# Diarization & Global speaker assignment
def diarize_chunks_with_global_ids_union(
    chunk_paths: Union[str, List[str]],
    attendee_num: int,
    overlap: float = CHUNK_OVERLAP,
    chunk_len: float = CHUNK_LENGTH,
    assign_threshold: float = 0.75,
    gap_tolerance: float = 0.3,
    force_with_kmeans: bool = True,   # NEW: enforce final #speakers with KMeans
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Run diarization per chunk, compute one embedding per local speaker (union with duration-weighted averaging),
    and assign global speaker IDs by cosine similarity against running centroids.
    Optionally, enforce the final number of speakers via KMeans reclustering.

    Parameters
    ----------
    chunk_paths : Union[str, List[str]]
        Either a directory path containing .wav chunks OR an explicit ordered list of chunk file paths.
    attendee_num : int
        Target number of speakers for diarization.
    overlap : float
        Overlap (s) used when splitting audio. Used to compute time offsets.
    chunk_len : float
        Chunk length (s).
    assign_threshold : float
        Cosine similarity threshold for incremental assignment.
    gap_tolerance : float
        Gap tolerance (s) for union within a local speaker.
    force_with_kmeans : bool
        If True and clusters > attendee_num, recluster to exactly attendee_num with KMeans.

    Returns
    -------
    all_segments : List[Dict]
        {start, end, speaker, chunk} with global speaker ids
    global_speakers : Dict[str, np.ndarray]
        Final centroids (if KMeans enforced, in normalized space)
    """
    # Check if models are loaded
    try:
        print(f"Checking diarization model...")
        diarization_model = models.diarization
        print(f"Diarization model loaded successfully")
        
        print(f"Checking inference model...")
        inference_model = models.inference
        print(f"Inference model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load required models: {e}")
        return [], {}
    
    # Normalize chunk list
    if isinstance(chunk_paths, str):
        chunk_files = sorted(
            [os.path.join(chunk_paths, f) for f in os.listdir(chunk_paths) if f.endswith('.wav')]
        )
    else:
        chunk_files = sorted(chunk_paths)

    print(f"Diarization: Found {len(chunk_files)} chunk files to process")
    if chunk_files:
        print(f"First few chunk files: {[os.path.basename(f) for f in chunk_files[:3]]}")
    else:
        print("ERROR: No chunk files found for diarization!")
        return [], {}

    all_segments: List[Dict] = []
    global_speakers: Dict[str, np.ndarray] = {}
    global_counts: Dict[str, int] = {}
    next_global_idx = 0

    for i, full_path in enumerate(chunk_files):
        offset = i * (chunk_len - overlap)
        print(f"Diarizing {os.path.basename(full_path)} with offset +{offset:.2f}s")
        
        # Check if file exists and has content
        if not os.path.exists(full_path):
            print(f"  ERROR: File {full_path} does not exist")
            continue
            
        file_size = os.path.getsize(full_path)
        print(f"  File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"  ERROR: File {full_path} is empty (0 bytes)")
            continue

        try:
            diarization = models.diarization(full_path, num_speakers=attendee_num)
            print(f"  Diarization model loaded successfully for {os.path.basename(full_path)}")
        except Exception as e:
            print(f"Diarization failed on {full_path}: {e}")
            continue

        local_turns: Dict[str, List[Segment]] = {}
        turn_count = 0
        for turn, _, local_label in diarization.itertracks(yield_label=True):
            local_turns.setdefault(local_label, []).append(turn)
            turn_count += 1
        
        print(f"  Found {turn_count} turns in {os.path.basename(full_path)}")
        print(f"  Local speaker labels: {list(local_turns.keys())}")
        
        # Check if no speech was detected
        if turn_count == 0:
            print(f"  WARNING: No speech turns detected in {os.path.basename(full_path)}")
            print(f"  This could mean:")
            print(f"    - The audio is silent")
            print(f"    - The audio quality is too poor for speech detection")
            print(f"    - The diarization model failed to detect speech")
            continue

        file_duration_sec = _audio_duration_seconds(full_path)
        print(f"  File duration: {file_duration_sec:.2f}s")
        
        # Check if file is too short or silent
        if file_duration_sec is not None and file_duration_sec < 0.1:
            print(f"  WARNING: File {os.path.basename(full_path)} is too short ({file_duration_sec:.2f}s), skipping")
            continue
        
        if file_duration_sec is not None and file_duration_sec < MIN_EMB_DUR:
            print(f"  WARNING: File {os.path.basename(full_path)} duration ({file_duration_sec:.2f}s) is shorter than minimum embedding duration ({MIN_EMB_DUR}s)")
            print(f"  This might cause issues with speaker detection")

        local_embs: Dict[str, np.ndarray] = {}
        audio_dict = {"audio": full_path}
        
        for local_label, turns in local_turns.items():
            merged = merge_segments(turns, gap_tolerance=gap_tolerance)
            print(f"  Speaker {local_label}: {len(turns)} turns -> {len(merged)} merged segments")
            
            vecs, durs = [], []
            for seg in merged:
                seg_for_emb = seg
                if seg_for_emb.duration < MIN_EMB_DUR:
                    if file_duration_sec is not None:
                        seg_for_emb = _ensure_min_duration(seg_for_emb, file_duration_sec)
                    if seg_for_emb.duration < MIN_EMB_DUR and file_duration_sec is None:
                        print(f"    Skipping segment {seg} - too short ({seg_for_emb.duration:.2f}s < {MIN_EMB_DUR}s)")
                        continue
                try:
                    emb = inference_model.crop(audio_dict, seg_for_emb)
                    emb = emb if isinstance(emb, np.ndarray) else np.asarray(emb)
                    print(f"    Successfully extracted embedding for segment {seg} (duration: {seg.duration:.2f}s)")
                except Exception as e:
                    print(f"    Failed to extract embedding for segment {seg}: {e}")
                    continue
                vecs.append(emb)
                durs.append(seg.duration)
            
            print(f"    Extracted {len(vecs)} embeddings for speaker {local_label}")
            if not vecs:
                print(f"    No valid embeddings for speaker {local_label}, skipping")
                continue
            local_embs[local_label] = duration_weighted_mean(vecs, durs)

        local_to_global: Dict[str, str] = {}
        print(f"  Processing {len(local_embs)} local speakers for global assignment")
        
        for local_label, local_vec in local_embs.items():
            best_gid, best_sim = None, -1.0
            for gid, centroid in global_speakers.items():
                sim = cosine_sim(local_vec, centroid)
                if sim > best_sim:
                    best_gid, best_sim = gid, sim

            if best_gid is not None and best_sim >= assign_threshold:
                gcount = global_counts.get(best_gid, 0)
                new_centroid = (global_speakers[best_gid] * gcount + local_vec) / (gcount + 1)
                global_speakers[best_gid] = new_centroid
                global_counts[best_gid] = gcount + 1
                local_to_global[local_label] = best_gid
                print(f"    Mapped local speaker {local_label} to existing global {best_gid} (sim: {best_sim:.3f})")
            else:
                gid = f"spk_{next_global_idx}"
                next_global_idx += 1
                global_speakers[gid] = local_vec
                global_counts[gid] = 1
                local_to_global[local_label] = gid
                print(f"    Created new global speaker {gid} for local {local_label}")

        print(f"  Local to global mapping: {local_to_global}")
        
        # Add segments to all_segments
        segment_count = 0
        for turn, _, local_label in diarization.itertracks(yield_label=True):
            gid = local_to_global.get(local_label)
            if gid is None:
                print(f"    Skipping turn for local speaker {local_label} - no global mapping")
                continue
            all_segments.append({
                "start": round(float(turn.start) + float(offset), 2),
                "end": round(float(turn.end) + float(offset), 2),
                "speaker": gid,
                "chunk": i,
            })
            segment_count += 1
        
        print(f"  Added {segment_count} segments from chunk {i}")

    # Optional KMeans enforcement
    if force_with_kmeans and len(global_speakers) > attendee_num:
        new_segments, new_centroids, mapping = force_num_speakers_kmeans(
            all_segments, global_speakers, attendee_num=attendee_num
        )
        new_centroids = {k: (np.asarray(v) if not isinstance(v, np.ndarray) else v) for k, v in new_centroids.items()}
        print(f"Diarization complete: {len(new_segments)} segments, {len(new_centroids)} speakers (KMeans enforced)")
        return new_segments, new_centroids

    print(f"Diarization complete: {len(all_segments)} segments, {len(global_speakers)} speakers")
    return all_segments, global_speakers
