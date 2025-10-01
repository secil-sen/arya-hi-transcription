"""
Default implementation of speaker diarization services.
"""

from typing import List, Dict, Any
import logging
from .transcript_service import SpeakerDiarizer

logger = logging.getLogger(__name__)


class DefaultSpeakerDiarizer(SpeakerDiarizer):
    """Default implementation of speaker diarizer."""

    async def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Perform speaker diarization using pyannote."""
        try:
            from app.core.model_registry import ModelRegistry
            from app.pipeline.diarization_utils import perform_diarization_with_overlap_resolution

            model_registry = ModelRegistry()
            diarization_pipeline = model_registry.get_diarization_model()

            if not diarization_pipeline:
                logger.warning("Diarization model not available, using single speaker")
                return self._create_single_speaker_segments(audio_path)

            segments = perform_diarization_with_overlap_resolution(
                audio_path, diarization_pipeline
            )

            logger.info(f"Diarization completed: {len(segments)} segments")
            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return self._create_single_speaker_segments(audio_path)

    def _create_single_speaker_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        """Create single speaker segments as fallback."""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)

            return [{
                "start": 0.0,
                "end": duration,
                "speaker": "SPEAKER_00",
                "text": ""
            }]
        except Exception:
            # Final fallback
            return [{
                "start": 0.0,
                "end": 60.0,
                "speaker": "SPEAKER_00",
                "text": ""
            }]