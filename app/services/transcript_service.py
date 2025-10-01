"""
Service layer for transcript processing following SOLID principles.
Single Responsibility: Each service has one clear purpose.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptRequest:
    """Data class for transcript request parameters."""
    audio_path: str
    user_id: str
    attendees: Optional[List[str]] = None
    meeting_id: Optional[str] = None
    output_root: str = "/tmp/transcripts"


@dataclass
class TranscriptResult:
    """Data class for transcript processing results."""
    success: bool
    transcript_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    processing_time_ms: Optional[float] = None


class TranscriptProcessor(ABC):
    """Abstract base class for transcript processors."""

    @abstractmethod
    async def process(self, request: TranscriptRequest) -> TranscriptResult:
        """Process transcript request."""
        pass


class AudioProcessor(ABC):
    """Abstract base class for audio processing."""

    @abstractmethod
    async def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video."""
        pass

    @abstractmethod
    async def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio to text."""
        pass


class SpeakerDiarizer(ABC):
    """Abstract base class for speaker diarization."""

    @abstractmethod
    async def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """Perform speaker diarization."""
        pass


class ContentEnricher(ABC):
    """Abstract base class for content enrichment."""

    @abstractmethod
    async def enrich(self, segments: List[Dict[str, Any]], attendees: List[str]) -> List[Dict[str, Any]]:
        """Enrich transcript content."""
        pass


class TranscriptService:
    """
    Main service orchestrating transcript processing.
    Dependency Injection: Accepts processors as dependencies.
    Open/Closed: Extensible without modification.
    """

    def __init__(
        self,
        audio_processor: AudioProcessor,
        speaker_diarizer: SpeakerDiarizer,
        content_enricher: ContentEnricher
    ):
        self.audio_processor = audio_processor
        self.speaker_diarizer = speaker_diarizer
        self.content_enricher = content_enricher

    async def process_transcript(self, request: TranscriptRequest) -> TranscriptResult:
        """
        Process complete transcript request.

        Args:
            request: Transcript processing request

        Returns:
            TranscriptResult: Processing result
        """
        try:
            start_time = self._get_current_time_ms()

            # Step 1: Validate request
            if not self._validate_request(request):
                return TranscriptResult(
                    success=False,
                    error_message="Invalid request parameters"
                )

            # Step 2: Setup working directory
            session_id = self._generate_session_id()
            work_dir = self._setup_working_directory(request, session_id)

            # Step 3: Extract audio
            audio_path = f"{work_dir}/audio.wav"
            audio_success = await self.audio_processor.extract_audio(
                request.audio_path, audio_path
            )
            if not audio_success:
                return TranscriptResult(
                    success=False,
                    error_message="Audio extraction failed"
                )

            # Step 4: Transcribe audio
            transcript_data = await self.audio_processor.transcribe(audio_path)

            # Step 5: Speaker diarization
            segments = await self.speaker_diarizer.diarize(audio_path)

            # Step 6: Generate dynamic attendees if needed
            attendees = self._resolve_attendees(request.attendees, segments)

            # Step 7: Content enrichment
            enriched_segments = await self.content_enricher.enrich(segments, attendees)

            # Step 8: Finalize result
            final_result = self._build_final_result(
                transcript_data, enriched_segments, attendees
            )

            processing_time = self._get_current_time_ms() - start_time

            return TranscriptResult(
                success=True,
                transcript_data=final_result,
                session_id=session_id,
                processing_time_ms=processing_time
            )

        except Exception as e:
            logger.error(f"Transcript processing failed: {e}")
            return TranscriptResult(
                success=False,
                error_message=str(e)
            )

    def _validate_request(self, request: TranscriptRequest) -> bool:
        """Validate transcript request."""
        return bool(request.audio_path and request.user_id)

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        from uuid import uuid4
        return uuid4().hex

    def _setup_working_directory(self, request: TranscriptRequest, session_id: str) -> str:
        """Setup working directory for processing."""
        from app.utils.path_utils import get_user_session_path
        return get_user_session_path(request.output_root, request.user_id, session_id)

    def _resolve_attendees(self, requested_attendees: Optional[List[str]], segments: List[Dict]) -> List[str]:
        """Resolve attendees list from request or generate dynamically."""
        if requested_attendees:
            return requested_attendees

        # Generate from detected speakers
        speakers = set()
        for segment in segments:
            speaker = segment.get('speaker', '').strip()
            if speaker and speaker not in ['UNKNOWN', 'unknown', '']:
                speakers.add(speaker)

        return [f"Speaker {i+1}" for i, _ in enumerate(sorted(speakers))] or ["Speaker 1", "Speaker 2"]

    def _build_final_result(
        self,
        transcript_data: Dict[str, Any],
        enriched_segments: List[Dict[str, Any]],
        attendees: List[str]
    ) -> Dict[str, Any]:
        """Build final result structure."""
        return {
            "transcript": transcript_data,
            "segments": enriched_segments,
            "attendees": attendees,
            "metadata": {
                "attendee_count": len(attendees),
                "segment_count": len(enriched_segments)
            }
        }

    def _get_current_time_ms(self) -> float:
        """Get current time in milliseconds."""
        import time
        return time.time() * 1000


# Factory pattern for creating service with default implementations
class TranscriptServiceFactory:
    """Factory for creating transcript service with appropriate processors."""

    @staticmethod
    def create_default_service() -> TranscriptService:
        """Create service with default processors."""
        from app.services.audio_processor_impl import DefaultAudioProcessor
        from app.services.diarizer_impl import DefaultSpeakerDiarizer
        from app.services.enricher_impl import DefaultContentEnricher

        return TranscriptService(
            audio_processor=DefaultAudioProcessor(),
            speaker_diarizer=DefaultSpeakerDiarizer(),
            content_enricher=DefaultContentEnricher()
        )