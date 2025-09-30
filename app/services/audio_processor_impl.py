"""
Default implementation of audio processing services.
"""

from typing import Dict, Any
import logging
from .transcript_service import AudioProcessor

logger = logging.getLogger(__name__)


class DefaultAudioProcessor(AudioProcessor):
    """Default implementation of audio processor."""

    async def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video using ffmpeg."""
        try:
            from app.pipeline.audio_utils import video_to_wav
            success = video_to_wav(video_path, output_path)
            logger.info(f"Audio extraction {'succeeded' if success else 'failed'}")
            return success
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False

    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        try:
            from app.core.model_registry import ModelRegistry
            import whisper_timestamped as whisper

            model_registry = ModelRegistry()
            whisper_model = model_registry.get_whisper_model()

            result = whisper.transcribe(
                whisper_model,
                audio_path,
                language="tr",
                detect_disfluencies=True
            )

            logger.info(f"Transcription completed: {len(result.get('segments', []))} segments")
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {}