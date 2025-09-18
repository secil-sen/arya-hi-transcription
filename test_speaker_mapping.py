#!/usr/bin/env python3
"""
Test script to verify the speaker name mapping functionality.
"""
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_speaker_mapping():
    """Test the apply_speaker_name_mapping function."""
    print("Testing speaker name mapping...")

    try:
        from app.pipeline.pipeline import apply_speaker_name_mapping

        # Create test segments with extracted names
        test_segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "spk_0",
                "text_corrected": "Ben İrem",
                "extracted_names": ["İrem"]
            },
            {
                "start": 5.0,
                "end": 10.0,
                "speaker": "spk_1",
                "text_corrected": "Merhaba Samet",
                "extracted_names": ["Samet"]
            },
            {
                "start": 10.0,
                "end": 15.0,
                "speaker": "spk_0",
                "text_corrected": "İrem sen ne düşünüyorsun?",
                "extracted_names": ["İrem"]
            },
            {
                "start": 15.0,
                "end": 20.0,
                "speaker": "spk_1",
                "text_corrected": "Samet burada mı?",
                "extracted_names": ["Samet"]
            },
            {
                "start": 20.0,
                "end": 25.0,
                "speaker": "spk_0",
                "text_corrected": "Evet, buradayım",
                "extracted_names": []
            }
        ]

        print(f"Input: {len(test_segments)} segments")
        print("  spk_0 appears in segments with names: ['İrem', 'İrem', []]")
        print("  spk_1 appears in segments with names: ['Samet', 'Samet']")

        # Apply mapping
        mapped_segments = apply_speaker_name_mapping(test_segments)

        print(f"\nOutput: {len(mapped_segments)} segments")

        # Check results
        spk_0_mapped = [seg["speaker"] for seg in mapped_segments if seg.get("original_speaker_id") == "spk_0"]
        spk_1_mapped = [seg["speaker"] for seg in mapped_segments if seg.get("original_speaker_id") == "spk_1"]

        print(f"spk_0 mapped to: {set(spk_0_mapped)}")
        print(f"spk_1 mapped to: {set(spk_1_mapped)}")

        # Verify mapping
        success = True
        if "İrem" not in spk_0_mapped:
            print("✗ spk_0 should be mapped to İrem")
            success = False
        if "Samet" not in spk_1_mapped:
            print("✗ spk_1 should be mapped to Samet")
            success = False

        if success:
            print("✓ Speaker mapping test passed!")
        else:
            print("✗ Speaker mapping test failed!")

        return success

    except Exception as e:
        print(f"✗ Speaker mapping test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases for speaker mapping."""
    print("\nTesting edge cases...")

    try:
        from app.pipeline.pipeline import apply_speaker_name_mapping

        # Test with no extracted names
        empty_segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "spk_0",
                "text_corrected": "Hello world",
                "extracted_names": []
            }
        ]

        mapped = apply_speaker_name_mapping(empty_segments)
        if mapped[0]["speaker"] == "spk_0":
            print("✓ Empty names case handled correctly")
        else:
            print("✗ Empty names case failed")
            return False

        # Test with conflicting names
        conflict_segments = [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "spk_0",
                "text_corrected": "Ben Ali",
                "extracted_names": ["Ali"]
            },
            {
                "start": 5.0,
                "end": 10.0,
                "speaker": "spk_0",
                "text_corrected": "Ben Veli",
                "extracted_names": ["Veli"]
            }
        ]

        mapped = apply_speaker_name_mapping(conflict_segments)
        # Should pick the most frequent (or first one if tied)
        mapped_name = mapped[0]["speaker"]
        if mapped_name in ["Ali", "Veli"]:
            print(f"✓ Conflict resolution handled: chose '{mapped_name}'")
        else:
            print(f"✗ Conflict resolution failed: got '{mapped_name}'")
            return False

        return True

    except Exception as e:
        print(f"✗ Edge case test failed: {e}")
        return False

def main():
    print("Speaker Name Mapping Test")
    print("=" * 30)

    success = True
    success &= test_speaker_mapping()
    success &= test_edge_cases()

    print("\n" + "=" * 30)
    if success:
        print("✓ All speaker mapping tests passed!")
        print("\nThe speaker name mapping should now:")
        print("  1. Extract names from segments correctly")
        print("  2. Map speaker IDs (spk_0, spk_1) to actual names")
        print("  3. Handle conflicts and edge cases gracefully")
        print("  4. Show actual names in the final transcript")
    else:
        print("❌ Some tests failed - please check the implementation")

if __name__ == "__main__":
    main()