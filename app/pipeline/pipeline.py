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

    try:
        wav_path = os.path.join(user_session_dir, "audio.wav")
        chunk_dir = os.path.join(user_session_dir, "chunks")
        output_json_path = os.path.join(user_session_dir, "transcript_V7.json")

        # Step 1: Convert and split
        convert_mp4_to_wav(mp4_path, wav_path)
        split_wav_into_chunks_v2(wav_path, chunk_dir)

        # Step 2: Diarization with offset
        diar_segments, speaker_map = diarize_chunks_with_global_ids_union(
            chunk_paths=chunk_dir,
            attendee_num=len(attendees),
        )

        # Step 3: Transcription with whisper
        transcribed_segment = run_whisper_transcription(diar_segments, chunk_dir)


        # Optional Step: Filter short segments:x
        filtered_segments = filter_short_segments(transcribed_segment)

        # Step 4: Correction and infer with Gemini
        aattendee_id_map = {
             "u_ufuk": "Ufuk",
             "u_samet": "Samet"
        }
        enriched_segments = asyncio.run(
            gemini_correct_and_infer_segments_async_v3(segments=filtered_segments,
                                                    attendee_list=attendees,
                                                    term_list=TERM_LIST,
                                                    attendee_id_map=aattendee_id_map,
                                                    enable_json_mode=True,
                                                    debug_dir="/Users/secilsen/Desktop/ARYA/code/transcript-api/tmp/gemini_debug")
        )
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
