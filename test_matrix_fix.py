#!/usr/bin/env python3
"""
Test script to verify the matrix operation fixes are working correctly.
"""
import numpy as np
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_cosine_sim():
    """Test the cosine similarity function with various inputs."""
    print("Testing cosine_sim function...")

    try:
        from app.pipeline.diarization import cosine_sim

        # Test normal case
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        sim = cosine_sim(a, b)
        print(f"✓ Normal case: {sim:.3f}")

        # Test shape mismatch
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5], dtype=np.float32)
        sim = cosine_sim(a, b)
        print(f"✓ Shape mismatch handled: {sim:.3f}")

        # Test with zero vectors
        a = np.array([0, 0, 0], dtype=np.float32)
        b = np.array([1, 2, 3], dtype=np.float32)
        sim = cosine_sim(a, b)
        print(f"✓ Zero vector handled: {sim:.3f}")

        return True
    except Exception as e:
        print(f"✗ cosine_sim test failed: {e}")
        return False

def test_duration_weighted_mean():
    """Test the duration weighted mean function."""
    print("\nTesting duration_weighted_mean function...")

    try:
        from app.pipeline.diarization import duration_weighted_mean

        # Test normal case
        vectors = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
            np.array([7, 8, 9], dtype=np.float32)
        ]
        durations = [1.0, 2.0, 1.0]
        result = duration_weighted_mean(vectors, durations)
        print(f"✓ Normal case: shape {result.shape}")

        # Test single vector
        vectors = [np.array([1, 2, 3], dtype=np.float32)]
        durations = [1.0]
        result = duration_weighted_mean(vectors, durations)
        print(f"✓ Single vector: shape {result.shape}")

        # Test different shapes (should be handled)
        vectors = [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5], dtype=np.float32)  # Different length
        ]
        durations = [1.0, 1.0]
        result = duration_weighted_mean(vectors, durations)
        print(f"✓ Different shapes handled: shape {result.shape}")

        return True
    except Exception as e:
        print(f"✗ duration_weighted_mean test failed: {e}")
        return False

def test_force_num_speakers_kmeans():
    """Test the KMeans clustering function."""
    print("\nTesting force_num_speakers_kmeans function...")

    try:
        from app.pipeline.diarization import force_num_speakers_kmeans

        # Test with None embeddings (fallback case)
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": "spk_0", "chunk": 0},
            {"start": 1.0, "end": 2.0, "speaker": "spk_1", "chunk": 0}
        ]
        global_speakers = {"spk_0": None, "spk_1": None}  # None embeddings
        attendee_num = 2

        new_segments, new_centroids, mapping = force_num_speakers_kmeans(
            segments, global_speakers, attendee_num
        )
        print(f"✓ None embeddings handled: {len(new_segments)} segments, {len(new_centroids)} speakers")

        # Test with valid embeddings
        global_speakers = {
            "spk_0": np.array([1, 2, 3], dtype=np.float32),
            "spk_1": np.array([4, 5, 6], dtype=np.float32)
        }

        new_segments, new_centroids, mapping = force_num_speakers_kmeans(
            segments, global_speakers, attendee_num
        )
        print(f"✓ Valid embeddings handled: {len(new_segments)} segments, {len(new_centroids)} speakers")

        return True
    except Exception as e:
        print(f"✗ force_num_speakers_kmeans test failed: {e}")
        return False

def main():
    print("Matrix Operations Fix Test")
    print("=" * 35)

    success = True

    success &= test_cosine_sim()
    success &= test_duration_weighted_mean()
    success &= test_force_num_speakers_kmeans()

    print("\n" + "=" * 35)
    if success:
        print("✓ All matrix operation tests passed!")
        print("\nThe matrix dimension fixes should resolve the matmul error.")
        print("Your transcription pipeline should now handle:")
        print("  1. Embedding shape mismatches gracefully")
        print("  2. None embeddings in KMeans clustering")
        print("  3. Vector dimension inconsistencies")
    else:
        print("❌ Some tests failed - please check the implementation")

if __name__ == "__main__":
    main()