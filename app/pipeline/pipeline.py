import os
import shutil
import requests
import json
from uuid import uuid4
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urlparse
from app.utils.path_utils import (
    get_user_session_path, get_audio_file_path, get_chunks_dir_path,
    get_debug_dir_path, safe_join
)

from typing import List, Optional
import asyncio

# 🚀 MONITORING IMPORT - Pipeline'da detaylı metrics için
from transcript_monitoring import transcript_monitor

from app.pipeline.ffmpeg_utils import convert_mp4_to_wav, split_wav_into_chunks_v2
from app.pipeline.diarization import diarize_chunks_with_global_ids_union
from app.pipeline.transcription import run_whisper_transcription
from app.pipeline.utils import save_as_json, filter_short_segments
from app.pipeline.correct_and_infer import gemini_correct_and_infer_segments_async_v3, enrich_names_with_gemini
from app.pipeline.config import DEFAULT_OUTPUT_ROOT, TERM_LIST
from app.pipeline.rule_based_name_extraction import apply_name_extraction_to_segments
from app.pipeline.embedding_extraction import extract_embeddings_async
from collections import Counter, defaultdict
try:
    from app.pipeline.notification import notify_chunking, save_as_jsonl
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    print("Warning: httpx not available, notification functionality disabled")


def _download_if_url(path: str, target_dir: str) -> str:
    """Downloads file if path is a URL. Supports both regular HTTP/HTTPS and S3 URLs."""
    if path.startswith("http://") or path.startswith("https://"):
        local_path = os.path.join(target_dir, "input.mp4")
        print(f"Downloading remote MP4 to {local_path} ...")

        # Check if this is an S3 URL
        parsed_url = urlparse(path)
        if '.s3.' in parsed_url.netloc and 'amazonaws.com' in parsed_url.netloc:
            try:
                # Extract bucket and key from S3 URL
                if parsed_url.netloc.startswith('s3.'):
                    # Format: https://s3.amazonaws.com/bucket/key or https://s3.region.amazonaws.com/bucket/key
                    bucket_name = parsed_url.path.split('/')[1]
                    object_key = '/'.join(parsed_url.path.split('/')[2:])
                else:
                    # Format: https://bucket.s3.region.amazonaws.com/key
                    bucket_name = parsed_url.netloc.split('.')[0]
                    object_key = parsed_url.path.lstrip('/')

                print(f"Detected S3 URL - Bucket: {bucket_name}, Key: {object_key}")

                # Initialize S3 client
                s3_client = boto3.client('s3')

                # Download file from S3
                print(f"Downloading from S3: s3://{bucket_name}/{object_key}")
                s3_client.download_file(bucket_name, object_key, local_path)
                print(f"S3 download finished: {local_path}")
                return local_path

            except NoCredentialsError:
                print("Error: AWS credentials not found. Please configure AWS credentials.")
                print("You can set them using:")
                print("- Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
                print("- AWS credentials file (~/.aws/credentials)")
                print("- IAM role (if running on EC2)")
                raise RuntimeError("AWS credentials not configured for S3 access")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchBucket':
                    raise RuntimeError(f"S3 bucket '{bucket_name}' does not exist")
                elif error_code == 'NoSuchKey':
                    raise RuntimeError(f"S3 object '{object_key}' does not exist in bucket '{bucket_name}'")
                elif error_code == 'AccessDenied':
                    raise RuntimeError(f"Access denied to S3 object. Check your AWS permissions for s3://{bucket_name}/{object_key}")
                else:
                    raise RuntimeError(f"S3 error: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to download from S3: {e}")
        else:
            # Regular HTTP/HTTPS download
            r = requests.get(path, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download finished: {local_path}")
            return local_path
    return path


def apply_speaker_name_mapping(segments):
    """
    Apply extracted names to speaker IDs based on co-occurrence and confidence.

    This function creates a mapping from speaker IDs (spk_0, spk_1, etc.) to
    actual names based on the extracted_names field in each segment.

    Args:
        segments: List of transcript segments with extracted_names field

    Returns:
        List of segments with speaker field updated to use actual names
    """
    if not segments:
        return segments

    # Count name occurrences per speaker
    speaker_name_counts = defaultdict(Counter)
    speaker_total_duration = defaultdict(float)

    # Collect statistics
    for seg in segments:
        speaker_id = seg.get("speaker", "")
        if not speaker_id or not speaker_id.startswith("spk_"):
            continue

        duration = seg.get("end", 0) - seg.get("start", 0)
        speaker_total_duration[speaker_id] += duration

        # Count extracted names for this speaker
        extracted_names = seg.get("extracted_names", [])
        for name in extracted_names:
            if name and name.strip():
                clean_name = name.strip()
                # Filter out generic terms and very short names
                if (len(clean_name) > 1 and
                    not clean_name.lower() in ['speaker', 'spk', 'unknown', 'speaker 1', 'speaker 2']):
                    speaker_name_counts[speaker_id][clean_name] += 1

    # Build speaker mapping
    speaker_mapping = {}
    assigned_names = set()

    print(f"Building speaker name mapping from {len(speaker_name_counts)} speakers...")

    # Sort speakers by total speaking duration (longer speakers get priority)
    sorted_speakers = sorted(speaker_total_duration.items(), key=lambda x: x[1], reverse=True)

    for speaker_id, duration in sorted_speakers:
        if speaker_id not in speaker_name_counts:
            continue

        name_counts = speaker_name_counts[speaker_id]
        if not name_counts:
            continue

        # Get most frequent name for this speaker
        most_common = name_counts.most_common()

        # Try to find a name that hasn't been assigned yet
        chosen_name = None
        for name, count in most_common:
            # Simple heuristics for name quality
            if (count >= 1 and  # At least mentioned once
                name not in assigned_names and  # Not already assigned
                len(name.split()) <= 3 and  # Not too long
                not any(char.isdigit() for char in name) and  # No numbers
                len(name) >= 2):  # At least 2 characters
                chosen_name = name
                break

        # If no name found with strict criteria, try looser criteria for common Turkish names
        if not chosen_name:
            for name, count in most_common:
                if (count >= 1 and
                    name not in assigned_names and
                    len(name) >= 2 and
                    # Check if it looks like a Turkish name (contains Turkish characters or common patterns)
                    (any(char in name for char in 'ÇĞİÖŞÜçğıöşü') or
                     name.lower() in ['irem', 'İrem', 'samet', 'ufuk', 'ahmet', 'mehmet', 'ali', 'veli', 'can', 'emre'])):
                    chosen_name = name
                    break

        if chosen_name:
            speaker_mapping[speaker_id] = chosen_name
            assigned_names.add(chosen_name)
            print(f"  {speaker_id} -> '{chosen_name}' (mentioned {name_counts[chosen_name]} times, {duration:.1f}s total)")
        else:
            print(f"  {speaker_id} -> no suitable name found (keeping original)")

    # Apply mapping to segments
    updated_segments = []
    for seg in segments:
        new_seg = seg.copy()
        speaker_id = seg.get("speaker", "")

        if speaker_id in speaker_mapping:
            new_seg["speaker"] = speaker_mapping[speaker_id]
            # Keep original speaker_id for reference
            new_seg["original_speaker_id"] = speaker_id

        updated_segments.append(new_seg)

    # Summary
    mapped_count = len(speaker_mapping)
    total_speakers = len(set(seg.get("speaker", "") for seg in segments if seg.get("speaker", "").startswith("spk_")))
    print(f"Speaker mapping complete: {mapped_count}/{total_speakers} speakers mapped to names")

    return updated_segments


# 🚀 DECORATOR EKLENDİ - Pipeline'da detaylı monitoring!
@transcript_monitor(
    user_context=lambda *args, **kwargs: kwargs.get('user_id', 'anonymous'),
    session_context=lambda *args, **kwargs: kwargs.get('meeting_id'),
    track_audio_metadata=True,
    custom_config={
        "track_performance_metrics": True,
        "track_quality_metrics": True,
        "offline_mode": True  # İlk test için offline
    }
)
async def run_transcript_pipeline(mp4_path: str, output_root: str = DEFAULT_OUTPUT_ROOT, user_id: str = "anonymous",
                                  attendees: Optional[List[str]] = None, meeting_id: Optional[str] = None) -> dict:
    """
    Video transcript pipeline - şimdi otomatik monitoring ile!

    Decorator otomatik olarak:
    - Pipeline'ın her aşamasının süresini ölçer
    - Audio file metadata'sını toplar
    - Quality metrics hesaplar
    - User ve session analytics yapar
    - Background'da Vertex AI'ya gönderir
    """
    session_id = uuid4().hex
    user_session_dir = get_user_session_path(output_root, user_id, session_id)

    # Debug input parameters
    print(f"Starting transcript pipeline:")
    print(f"- MP4 path: {mp4_path}")
    print(f"- Output root: {output_root}")
    print(f"- User ID: {user_id}")
    print(f"- Session ID: {session_id}")
    print(f"- Meeting ID: {meeting_id}")
    print(f"- Attendees: {attendees}")
    print(f"- User session dir: {user_session_dir}")

    # Handle attendees list - prefer provided list, fallback to dynamic generation
    if not attendees:
        print("INFO: No attendees provided, will use dynamic speaker detection")
        attendees = []  # Empty list will trigger dynamic detection

    print(f"Using attendees: {attendees if attendees else 'Dynamic detection'}")

    try:
        wav_path = get_audio_file_path(user_session_dir, "audio.wav")
        chunk_dir = get_chunks_dir_path(user_session_dir)
        output_json_path = safe_join(user_session_dir, "transcript_V7.json")

        # NEW: Eğer mp4_path URL ise indir
        mp4_path = _download_if_url(mp4_path, user_session_dir)

        # Step 1: Convert and split
        convert_mp4_to_wav(mp4_path, wav_path)
        print(f"Audio converted to WAV: {wav_path}")

        split_wav_into_chunks_v2(wav_path, chunk_dir)
        print(f"WAV split into chunks in directory: {chunk_dir}")

        # Check chunk files
        chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.wav')]
        print(f"Created {len(chunk_files)} chunk files: {chunk_files[:5]}...")  # Show first 5

        # Step 2: Diarization with offset (optional if using Replicate with diarization)
        use_replicate_transcription = (
            os.getenv("REPLICATE_API_TOKEN") and
            os.getenv("REPLICATE_API_TOKEN") != "r8_xxx..." and
            os.getenv("USE_REPLICATE_DIARIZATION", "true").lower() == "true"
        )

        if use_replicate_transcription:
            print(" Using Replicate with integrated diarization - skipping separate diarization step")
            # Create minimal segments for Replicate transcription
            diar_segments = [{
                "start": 0.0,
                "end": 240.0,  # Will be updated by Replicate
                "speaker": "spk_0",
                "chunk": 0
            }]
            speaker_map = {}
        else:
            print("🔄 Using traditional diarization pipeline")
            diar_segments, speaker_map = diarize_chunks_with_global_ids_union(
                chunk_paths=chunk_dir,
                attendee_num=len(attendees),
            )

        print(f"Diarization segments count: {len(diar_segments)}")
        if diar_segments:
            print(f"First diarization segment: {diar_segments[0]}")
        else:
            print("WARNING: No segments from diarization!")

        # Step 3: Transcription with whisper
        transcribed_segment = run_whisper_transcription(diar_segments, chunk_dir)
        print(f"Transcribed segments count: {len(transcribed_segment)}")

        # Optional Step: Filter short segments
        filtered_segments = filter_short_segments(transcribed_segment)
        print(f"Filtered segments count: {len(filtered_segments)}")

        if filtered_segments:
            print(f"First filtered segment: {filtered_segments[0]}")
        else:
            print("WARNING: No segments after filtering!")

        # Step 4: Generate dynamic attendees if none provided
        if not attendees and filtered_segments:
            # Extract unique speakers from diarization results
            unique_speakers = set()
            for segment in filtered_segments:
                speaker = segment.get('speaker', '').strip()
                if speaker and speaker not in ['UNKNOWN', 'unknown', '']:
                    unique_speakers.add(speaker)

            # Generate attendee names based on detected speakers
            attendees = [f"Speaker {i+1}" for i, _ in enumerate(sorted(unique_speakers))]
            if len(attendees) == 0:
                attendees = ["Speaker 1", "Speaker 2"]  # Fallback

            print(f"Generated dynamic attendees: {attendees}")

        # Step 5: Correction and infer with Gemini
        attendee_id_map = {f"u_{attendee.lower()}": attendee for attendee in attendees}

        debug_dir = get_debug_dir_path(user_session_dir, "gemini_debug") if os.getenv("DEBUG_MODE",
                                                                                "false").lower() == "true" else None

        # 🚀 ASYNC FIX - Pipeline artık async, direkt await kullan
        enriched_segments = await gemini_correct_and_infer_segments_async_v3(
            segments=filtered_segments,
            attendee_list=attendees,
            term_list=TERM_LIST,
            attendee_id_map=attendee_id_map,
            enable_json_mode=True,
            debug_dir=debug_dir
        )

        print(f"Enriched segments count: {len(enriched_segments)}")
        if enriched_segments:
            print(f"First enriched segment: {enriched_segments[0]}")
        else:
            print("WARNING: No segments after Gemini processing!")

        # Step 5: Apply rule-based name extraction (NER)
        print("Applying rule-based name extraction (NER)...")
        ner_start_time = __import__('time').time()

        try:
            # Only apply name extraction if we have segments and attendees
            if enriched_segments and attendees:
                # NER Configuration - Using bert-base model (much safer than xlm-roberta)
                # Your selected model: "Davlan/bert-base-multilingual-cased-ner-hrl"
                # Heavy model (avoid): "Davlan/xlm-roberta-base-ner-hrl"
                ner_model = "Davlan/bert-base-multilingual-cased-ner-hrl"
                print(f"🤖 NER MODEL ENABLED: {ner_model}")
                print("   Using BERT-base model (safer than XLM-RoBERTa)")

                enriched_segments = apply_name_extraction_to_segments(
                    segments=enriched_segments,
                    attendees=attendees,
                    ner_model=ner_model,
                    tau_ms=90000,  # 90 seconds temporal window
                    threshold=0.4,  # Minimum confidence threshold
                    spk_strong_thr=1.5,  # Speaker canonical assignment threshold
                    spk_margin=0.6,  # Margin between top candidates
                    spk_min_dur_ms=3000  # Minimum speaker duration (3 seconds)
                )

                # Count segments with extracted names
                segments_with_names = sum(1 for seg in enriched_segments if seg.get("extracted_names"))
                total_names = sum(len(seg.get("extracted_names", [])) for seg in enriched_segments)

                ner_elapsed_time = __import__('time').time() - ner_start_time
                print(f"✅ Name extraction (NER) completed in {ner_elapsed_time:.2f}s")
                print(f"   - {segments_with_names}/{len(enriched_segments)} segments have extracted names")
                print(f"   - Total names extracted: {total_names}")

                if enriched_segments and enriched_segments[0].get("extracted_names"):
                    print(f"   - Sample names (first segment): {enriched_segments[0].get('extracted_names', [])}")

                # Debug: Show all extracted names for verification
                print("   - All extracted names by segment:")
                for i, seg in enumerate(enriched_segments[:5]):  # Show first 5 segments
                    names = seg.get('extracted_names', [])
                    if names:
                        print(f"     Segment {i+1}: {names}")
                    else:
                        print(f"     Segment {i+1}: (no names)")
                if len(enriched_segments) > 5:
                    print(f"     ... and {len(enriched_segments) - 5} more segments")
            else:
                print("⚠️  Skipping name extraction: No segments or attendees provided")
                # Add empty extracted_names field to all segments
                for seg in enriched_segments:
                    seg["extracted_names"] = []

        except ImportError as e:
            print(f"⚠️  Name extraction dependency missing: {e}")
            print("   Installing rapidfuzz: pip install rapidfuzz")
            # Add empty extracted_names field to all segments
            for seg in enriched_segments:
                seg["extracted_names"] = []
        except Exception as e:
            print(f"⚠️  Name extraction failed: {e}")
            # Continue pipeline even if name extraction fails
            for seg in enriched_segments:
                seg["extracted_names"] = []

        # Step 5.5: Correct extracted names with Gemini
        print("Correcting extracted names with Gemini...")
        try:
            enriched_segments = await enrich_names_with_gemini(enriched_segments)
        except Exception as e:
            print(f"⚠️  Name correction with Gemini failed: {e}")
            # Continue pipeline even if name correction fails
            pass

        # Step 5.6: Extract embeddings from audio segments
        print("Extracting speaker embeddings from audio...")
        embedding_start_time = __import__('time').time()

        try:
            from app.core.model_registry import models

            # Extract embeddings using the WAV file
            enriched_segments = await extract_embeddings_async(
                segments=enriched_segments,
                audio_path=wav_path,
                inference_model=models.inference
            )

            # Count segments with embeddings
            segments_with_embeddings = sum(1 for seg in enriched_segments if seg.get("embedding") is not None)
            embedding_elapsed_time = __import__('time').time() - embedding_start_time

            print(f"✅ Embedding extraction completed in {embedding_elapsed_time:.2f}s")
            print(f"   - {segments_with_embeddings}/{len(enriched_segments)} segments have embeddings")

            if segments_with_embeddings > 0:
                sample_dim = enriched_segments[0].get("embedding_dim", "unknown")
                print(f"   - Embedding dimension: {sample_dim}")

        except Exception as e:
            print(f"⚠️  Embedding extraction failed: {e}")
            print("   Continuing pipeline without embeddings...")
            # Add None embeddings to all segments
            for seg in enriched_segments:
                if "embedding" not in seg:
                    seg["embedding"] = None
                    seg["embedding_error"] = str(e)

        # Step 6: Save output as JSONL (if notification available)
        output_jsonl_path = os.path.join(user_session_dir, "transcript.jsonl")
        if NOTIFICATION_AVAILABLE:
            save_as_jsonl(enriched_segments, output_jsonl_path)
        else:
            # Fallback: save as JSONL manually
            with open(output_jsonl_path, "w", encoding="utf-8") as f:
                for item in enriched_segments:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Also save as JSON for backward compatibility
        save_as_json(enriched_segments, output_json_path)

        # Step 7: Send notification to chunking service
        notification_result = None
        if meeting_id and NOTIFICATION_AVAILABLE:
            try:
                chunking_url = os.getenv("CHUNKING_SERVICE_URL", "http://localhost:8090")
                notification_result = notify_chunking(
                    meeting_id=meeting_id,
                    transcript_path=output_jsonl_path,
                    chunking_url=chunking_url
                )
                print(f"Notification sent successfully: {notification_result}")
            except Exception as e:
                print(f"Warning: Failed to send notification to chunking service: {e}")
                # Don't fail the entire pipeline if notification fails
        elif meeting_id and not NOTIFICATION_AVAILABLE:
            print("Warning: httpx not available, notification skipped")
        elif not meeting_id:
            print("Warning: meeting_id not provided, skipping notification")

        # Step 7: Apply extracted names to speaker IDs
        print("Applying extracted names to speaker IDs...")
        enriched_segments = apply_speaker_name_mapping(enriched_segments)

        # 8. Clean up (if not debug)
        if os.getenv("DEBUG_MODE", "false").lower() != "true":
            shutil.rmtree(user_session_dir)

        return {
            "status": "success",
            "session_id": session_id,
            "segments": enriched_segments,
            "path": output_json_path,
            "jsonl_path": output_jsonl_path,
            "notification_result": notification_result
        }
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}")


if __name__ == '__main__':
    run_transcript_pipeline(
        "https://api-lia.arya-ai.com/api/public/meetings/1/af8f6828-6f9c-4331-afb4-8a3ae40349de_20250827_081821.mp4",
        user_id="deb",
        attendees=["Ufuk", "Samet"]
    )
