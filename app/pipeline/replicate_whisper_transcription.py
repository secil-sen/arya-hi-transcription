import os
import time
from typing import List, Dict, Any, Optional
from app.core.model_registry import models
from app.pipeline.replicate_transcription import transcribe_with_replicate


def run_replicate_whisper_transcription(
    wav_file_path: str,
    language: str = "tr",
    enable_diarization: bool = True,
    enable_fallback: bool = True
) -> List[Dict[str, Any]]:
    """
    Transcribe a WAV file using Replicate's Incredibly Fast Whisper model.

    This function transcribes an entire WAV file (not segment-based like the original)
    and returns segments compatible with the existing pipeline format.

    Parameters
    ----------
    wav_file_path : str
        Path to the WAV file to transcribe
    language : str
        Language code (default: "tr" for Turkish)
    enable_diarization : bool
        Enable speaker diarization (default: True)
    enable_fallback : bool
        Fall back to local Whisper if Replicate fails (default: True)

    Returns
    -------
    List[Dict] : Each item contains {start, end, speaker, text}
    """
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"WAV file not found: {wav_file_path}")

    print(f"🚀 Starting Replicate transcription for: {os.path.basename(wav_file_path)}")
    start_time = time.time()

    try:
        # Use Replicate transcription service
        transcription_result = transcribe_with_replicate(
            audio_file_path=wav_file_path,
            language=language,
            enable_diarization=enable_diarization
        )

        elapsed_time = time.time() - start_time
        print(f"✅ Replicate transcription completed in {elapsed_time:.2f} seconds")

        # Parse Replicate response into segments
        segments = parse_replicate_response(transcription_result)

        print(f"📊 Transcription results:")
        print(f"   - Total segments: {len(segments)}")
        if segments:
            print(f"   - Duration: {segments[0].get('start', 0):.2f}s - {segments[-1].get('end', 0):.2f}s")
            print(f"   - Sample segment: {segments[0]}")

        return segments

    except Exception as e:
        print(f"❌ Replicate transcription failed: {e}")

        if enable_fallback:
            print(f"🔄 Falling back to local Whisper transcription...")
            return run_fallback_whisper_transcription(wav_file_path, language)
        else:
            raise RuntimeError(f"Replicate transcription failed and fallback disabled: {e}")


def parse_replicate_response(transcription_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse Replicate transcription response into the expected segment format.

    Expected Replicate format:
    {
        "text": "Full transcript text",
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Segment text",
                "speaker": "SPEAKER_00"  # if diarization enabled
            }
        ]
    }

    Returns segments in format: {start, end, speaker, text}
    """
    segments = []

    # Get segments from Replicate response
    replicate_segments = transcription_result.get("segments", [])

    if not replicate_segments:
        # If no segments, create one segment from full text
        full_text = transcription_result.get("text", "").strip()
        if full_text:
            segments.append({
                "start": 0.0,
                "end": 10.0,  # Default duration
                "speaker": "SPEAKER_00",
                "text": full_text
            })
        return segments

    # Process each segment
    for i, segment in enumerate(replicate_segments):
        try:
            start_time = float(segment.get("start", 0.0))
            end_time = float(segment.get("end", start_time + 1.0))
            text = str(segment.get("text", "")).strip()

            # Handle speaker ID
            speaker = segment.get("speaker", f"SPEAKER_{i:02d}")

            # Convert Replicate speaker format to our format
            if speaker.startswith("SPEAKER_"):
                # Convert SPEAKER_00, SPEAKER_01 to spk_0, spk_1
                speaker_num = speaker.split("_")[-1]
                try:
                    speaker_id = int(speaker_num)
                    speaker = f"spk_{speaker_id}"
                except (ValueError, IndexError):
                    speaker = f"spk_{i % 2}"  # Fallback

            if text:  # Only add segments with text
                segments.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "speaker": speaker,
                    "text": text
                })

        except Exception as e:
            print(f"⚠️  Warning: Failed to parse segment {i}: {e}")
            continue

    print(f"✅ Parsed {len(segments)} segments from Replicate response")
    return segments


def run_fallback_whisper_transcription(wav_file_path: str, language: str = "tr") -> List[Dict[str, Any]]:
    """
    Fallback to local Whisper transcription if Replicate fails.

    This uses the original Whisper model for transcription as a backup.
    """
    print(f"🔄 Using fallback Whisper transcription...")

    try:
        # Use local Whisper model
        whisper_model = models.whisper
        if whisper_model is None:
            raise RuntimeError("Local Whisper model not available")

        start_time = time.time()

        # Transcribe with local Whisper
        result = whisper_model.transcribe(wav_file_path, language=language)

        elapsed_time = time.time() - start_time
        print(f"✅ Fallback Whisper completed in {elapsed_time:.2f} seconds")

        # Parse Whisper response
        if isinstance(result, dict) and "segments" in result:
            segments = []
            for i, segment in enumerate(result["segments"]):
                segments.append({
                    "start": round(float(segment.get("start", 0)), 2),
                    "end": round(float(segment.get("end", 0)), 2),
                    "speaker": f"spk_{i % 2}",  # Simple alternating speakers
                    "text": str(segment.get("text", "")).strip()
                })
            return segments
        else:
            # Simple text result
            text = str(result).strip() if result else ""
            if text:
                return [{
                    "start": 0.0,
                    "end": 10.0,
                    "speaker": "spk_0",
                    "text": text
                }]
            else:
                return []

    except Exception as e:
        print(f"❌ Fallback Whisper also failed: {e}")
        raise RuntimeError(f"Both Replicate and fallback Whisper transcription failed: {e}")


def run_replicate_whisper_transcription_for_segments(
    segments: List[Dict[str, Any]],
    chunk_dir: str,
    language: str = "tr"
) -> List[Dict[str, Any]]:
    """
    Compatibility function for the existing pipeline that expects segment-based transcription.

    This function maintains compatibility with the existing pipeline structure
    but uses Replicate for the actual transcription instead of processing individual segments.

    Parameters
    ----------
    segments : List[Dict]
        Diarization segments (used for reference, but transcription is done on full audio)
    chunk_dir : str
        Directory containing chunk files
    language : str
        Language code

    Returns
    -------
    List[Dict] : Transcribed segments
    """
    print(f"🔗 Compatibility mode: Using Replicate for segment-based transcription")

    # Check if we should use integrated diarization
    use_integrated_diarization = os.getenv("USE_REPLICATE_DIARIZATION", "true").lower() == "true"

    # Find all chunk files
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.wav')]
    if not chunk_files:
        print(f"❌ No chunk files found in {chunk_dir}")
        return []

    # Sort chunk files
    chunk_files.sort()

    all_segments = []
    chunk_offset = 0.0

    # Get chunk configuration
    chunk_length = float(os.getenv("CHUNK_LENGTH", "240"))
    chunk_overlap = float(os.getenv("CHUNK_OVERLAP", "4"))
    chunk_hop = chunk_length - chunk_overlap

    for chunk_file in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunk_file)

        try:
            # Transcribe this chunk with Replicate
            chunk_segments = run_replicate_whisper_transcription(
                wav_file_path=chunk_path,
                language=language,
                enable_diarization=use_integrated_diarization,
                enable_fallback=True
            )

            # Adjust timestamps to account for chunk offset
            for segment in chunk_segments:
                segment["start"] = round(segment["start"] + chunk_offset, 2)
                segment["end"] = round(segment["end"] + chunk_offset, 2)
                # Add chunk information for compatibility
                segment["chunk"] = len(all_segments) // 10  # Rough estimate
                all_segments.append(segment)

            # Update offset for next chunk
            chunk_offset += chunk_hop

        except Exception as e:
            print(f"⚠️  Warning: Failed to transcribe chunk {chunk_file}: {e}")
            continue

    print(f"✅ Processed {len(chunk_files)} chunks, got {len(all_segments)} total segments")

    # If using integrated diarization, ensure consistent speaker naming
    if use_integrated_diarization and all_segments:
        all_segments = normalize_speaker_ids(all_segments)

    return all_segments


def normalize_speaker_ids(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize speaker IDs across all segments to ensure consistency.
    Convert various speaker formats to spk_0, spk_1, etc.
    """
    print("🎯 Normalizing speaker IDs across segments...")

    # Track speaker mappings
    speaker_mapping = {}
    next_speaker_id = 0

    for segment in segments:
        original_speaker = segment.get("speaker", "unknown")

        # Map speaker to normalized ID
        if original_speaker not in speaker_mapping:
            speaker_mapping[original_speaker] = f"spk_{next_speaker_id}"
            next_speaker_id += 1

        # Update segment with normalized speaker ID
        segment["speaker"] = speaker_mapping[original_speaker]

    print(f"✅ Normalized {len(speaker_mapping)} unique speakers: {list(speaker_mapping.values())}")
    return segments