import os
import shutil
import requests
import json
from uuid import uuid4
import asyncio

from typing import List, Optional

from app.pipeline.ffmpeg_utils import convert_mp4_to_wav, split_wav_into_chunks_v2
from app.pipeline.diarization import diarize_chunks_with_global_ids_union
from app.pipeline.transcription import run_whisper_transcription
from app.pipeline.utils import save_as_json, filter_short_segments
from app.pipeline.correct_and_infer import gemini_correct_and_infer_segments_async_v3
from app.pipeline.config import DEFAULT_OUTPUT_ROOT, TERM_LIST
from app.pipeline.rule_based_name_extraction import apply_name_extraction_to_segments
try:
    from app.pipeline.notification import notify_chunking, save_as_jsonl
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    print("Warning: httpx not available, notification functionality disabled")


def _download_if_url(path: str, target_dir: str) -> str:
    """Eğer path bir http/https URL ise önce indirip local path döndürür."""
    if path.startswith("http://") or path.startswith("https://"):
        local_path = os.path.join(target_dir, "input.mp4")
        print(f"Downloading remote MP4 to {local_path} ...")
        r = requests.get(path, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download finished: {local_path}")
        return local_path
    return path


def run_transcript_pipeline(mp4_path: str, output_root: str = DEFAULT_OUTPUT_ROOT, user_id: str = "anonymous",
                            attendees: Optional[List[str]] = None, meeting_id: Optional[str] = None) -> dict:
    session_id = uuid4().hex
    user_session_dir = os.path.join(output_root, user_id, session_id)
    os.makedirs(user_session_dir, exist_ok=True)

    # Debug input parameters
    print(f"Starting transcript pipeline:")
    print(f"- MP4 path: {mp4_path}")
    print(f"- Output root: {output_root}")
    print(f"- User ID: {user_id}")
    print(f"- Session ID: {session_id}")
    print(f"- Meeting ID: {meeting_id}")
    print(f"- Attendees: {attendees}")
    print(f"- User session dir: {user_session_dir}")

    # Safety check for attendees
    if not attendees:
        print("WARNING: No attendees provided, using default")
        attendees = ["Speaker1", "Speaker2"]  # Default attendees

    print(f"Using attendees: {attendees}")

    try:
        wav_path = os.path.join(user_session_dir, "audio.wav")
        chunk_dir = os.path.join(user_session_dir, "chunks")
        output_json_path = os.path.join(user_session_dir, "transcript_V7.json")

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

        # Step 2: Diarization with offset
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

        # Step 4: Correction and infer with Gemini
        attendee_id_map = {f"u_{attendee.lower()}": attendee for attendee in attendees}

        debug_dir = os.path.join(user_session_dir, "gemini_debug") if os.getenv("DEBUG_MODE",
                                                                                "false").lower() == "true" else None

        enriched_segments = asyncio.run(
            gemini_correct_and_infer_segments_async_v3(
                segments=filtered_segments,
                attendee_list=attendees,
                term_list=TERM_LIST,
                attendee_id_map=attendee_id_map,
                enable_json_mode=True,
                debug_dir=debug_dir
            )
        )

        print(f"Enriched segments count: {len(enriched_segments)}")
        if enriched_segments:
            print(f"First enriched segment: {enriched_segments[0]}")
        else:
            print("WARNING: No segments after Gemini processing!")

        # Step 5: Apply rule-based name extraction
        print("Applying rule-based name extraction...")
        start_time = __import__('time').time()
        
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
                
                elapsed_time = __import__('time').time() - start_time
                print(f"✅ Name extraction completed in {elapsed_time:.2f}s")
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
