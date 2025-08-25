import os
import shutil
from uuid import uuid4
import asyncio

from typing import List, Optional

from app.pipeline.ffmpeg_utils import convert_mp4_to_wav, split_wav_into_chunks_v2
from app.pipeline.diarization import diarize_chunks_with_global_ids_union
from app.pipeline.transcription import run_whisper_transcription
from app.pipeline.utils import save_as_json, filter_short_segments
from app.pipeline.correct_and_infer import gemini_correct_and_infer_segments_async_v3
from app.pipeline.config import DEFAULT_OUTPUT_ROOT, TERM_LIST

# .env dosyasını yükle

def run_transcript_pipeline(mp4_path: str, output_root: str = DEFAULT_OUTPUT_ROOT, user_id: str = "anonymous",
                            attendees: Optional[List[str]] = None) -> dict:
    session_id = uuid4().hex
    user_session_dir = os.path.join(output_root, user_id, session_id)
    os.makedirs(user_session_dir, exist_ok=True)

    # Debug input parameters
    print(f"Starting transcript pipeline:")
    print(f"- MP4 path: {mp4_path}")
    print(f"- Output root: {output_root}")
    print(f"- User ID: {user_id}")
    print(f"- Session ID: {session_id}")
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
        # Create attendee ID mapping dynamically
        attendee_id_map = {}
        for i, attendee in enumerate(attendees):
            attendee_id_map[f"u_{attendee.lower()}"] = attendee
        
        # Create debug directory in the user session directory
        debug_dir = os.path.join(user_session_dir, "gemini_debug") if os.getenv("DEBUG_MODE", "false").lower() == "true" else None
        
        enriched_segments = asyncio.run(
            gemini_correct_and_infer_segments_async_v3(segments=filtered_segments,
                                                    attendee_list=attendees,
                                                    term_list=TERM_LIST,
                                                    attendee_id_map=attendee_id_map,
                                                    enable_json_mode=True,
                                                    debug_dir=debug_dir)
        )
        
        print(f"Enriched segments count: {len(enriched_segments)}")
        if enriched_segments:
            print(f"First enriched segment: {enriched_segments[0]}")
        else:
            print("WARNING: No segments after Gemini processing!")
        
        # Step 6: Save output
        save_as_json(enriched_segments, output_json_path)

        # 7. Clean up (if not debug)
        if os.getenv("DEBUG_MODE", "false").lower() != "true":
            shutil.rmtree(user_session_dir)

        return {
            "status": "success",
            "session_id": session_id,
            "segments": enriched_segments,
            "path": output_json_path
        }
    except Exception as e:
        raise RuntimeError(f"Pipeline failed: {e}")

if __name__ == '__main__':
    run_transcript_pipeline("",
                            "",
                            user_id="deb",
                            attendees=["Ufuk", "Samet"])
