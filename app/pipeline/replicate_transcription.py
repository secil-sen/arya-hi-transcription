import os
import asyncio
import time
from typing import List, Dict, Any, Optional
import replicate
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import urlparse
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env file is loaded before checking environment variables
load_dotenv()


class ReplicateTranscriptionService:
    """Replicate-based transcription service using WhisperX model."""

    def __init__(self):
        self.client = None
        # Updated to victor-upmeet/whisperx model with pyannote diarization and word-level timestamps
        self.model_version = "victor-upmeet/whisperx:84d2ad2d6194fe98a17d2b60bef1c7f910c46b2f6fd38996ca457afd9c8abfcb"
        self._initialized = False

    def _initialize_client(self):
        """Initialize Replicate client with API token (lazy loading)."""
        if self._initialized:
            return

        # Reload .env in case it wasn't loaded during import
        load_dotenv()

        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise EnvironmentError(
                "REPLICATE_API_TOKEN not found in environment variables. "
                "Please set it in your .env file."
            )

        if api_token == "r8_xxx...":
            raise EnvironmentError(
                "Please replace 'r8_xxx...' with your actual Replicate API token "
                "in the .env file. Get your token from: https://replicate.com/account/api-tokens"
            )

        # Set the API token for replicate client
        os.environ["REPLICATE_API_TOKEN"] = api_token
        self._initialized = True
        print(f"✅ Replicate client initialized with model: {self.model_version}")

    def _create_s3_presigned_url(self, file_path: str, expiration: int = 3600) -> Optional[str]:
        """Create a pre-signed URL for S3 access to the audio file."""
        try:
            # Create S3 client
            s3_client = boto3.client('s3')

            # Use the existing bucket from the environment if available
            bucket_name = os.getenv("AWS_S3_BUCKET", "lia-meeting-files")
            object_key = f"temp-transcription/{int(time.time())}_{os.path.basename(file_path)}"

            try:
                # Try to upload the file to S3
                s3_client.upload_file(file_path, bucket_name, object_key)

                # Create pre-signed URL
                presigned_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': object_key},
                    ExpiresIn=expiration
                )

                print(f"✅ Created S3 pre-signed URL for {os.path.basename(file_path)}")
                return presigned_url

            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code in ['NoSuchBucket', 'AccessDenied']:
                    print(f"⚠️  S3 bucket access issue ({error_code}), using direct file upload")
                else:
                    print(f"⚠️  S3 upload failed: {e}")
                return None

        except Exception as e:
            print(f"⚠️  Failed to create S3 pre-signed URL: {e}")
            return None

    async def transcribe_audio_async(
        self,
        audio_file_path: str,
        language: str = "tr",
        batch_size: int = 24,
        return_timestamps: bool = True,
        enable_diarization: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Replicate's Whisper Diarization model.

        Args:
            audio_file_path: Path to the audio file (WAV format recommended)
            language: Language code (default: "tr" for Turkish)
            batch_size: Not used in new model (kept for compatibility)
            return_timestamps: Not used in new model (kept for compatibility)
            enable_diarization: Whether to enable speaker diarization

        Returns:
            Dict containing transcription results with segments and text
        """
        # Initialize client on first use
        self._initialize_client()
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Get HuggingFace token from environment (required for WhisperX diarization)
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN not found in environment variables. "
                "Required for WhisperX diarization. Please set it in your .env file."
            )

        # Try to create S3 pre-signed URL first (fastest option)
        s3_url = self._create_s3_presigned_url(audio_file_path)

        # Prepare input parameters for WhisperX model
        input_params = {
            "language": language,  # ISO language code (e.g., "tr" for Turkish)
            "diarization": True,  # Enable speaker diarization with pyannote
            "align_output": True,  # Enable word-level timestamps
            "huggingface_access_token": hf_token,  # Required for pyannote diarization
            "batch_size": 64,  # Default batch size
            "temperature": 0,  # Sampling temperature
            "vad_onset": 0.5,  # Voice activity detection onset
            "vad_offset": 0.363,  # Voice activity detection offset
            "initial_prompt": "",  # Empty initial prompt
            "language_detection_min_prob": 0,  # Language detection probability threshold
            "language_detection_max_tries": 5,  # Max language detection attempts
            "debug": False  # Disable debug output
            # min_speakers: omitted (auto-detect)
            # max_speakers: omitted (auto-detect)
        }

        # Use S3 URL if available, otherwise open file directly
        if s3_url:
            input_params["audio_file"] = s3_url
            print(f"📡 Using S3 URL for audio input: {os.path.basename(audio_file_path)}")
        else:
            # Fallback to file upload
            with open(audio_file_path, "rb") as audio_file:
                input_params["audio_file"] = audio_file
                print(f"📁 Using direct file upload: {os.path.basename(audio_file_path)}")

        try:
            print(f"🚀 Starting Replicate WhisperX transcription...")
            print(f"   Model: victor-upmeet/whisperx")
            print(f"   Language: {language}")
            print(f"   Diarization: {input_params['diarization']}")
            print(f"   Align output (word-level): {input_params['align_output']}")
            print(f"   Auto-detect speakers: True (min/max omitted)")

            start_time = time.time()

            # Run prediction
            if s3_url:
                # Async prediction for URL input
                prediction = replicate.predictions.create(
                    version=self.model_version,
                    input=input_params
                )

                # Wait for completion
                print(f"⏳ Waiting for transcription to complete...")
                prediction = replicate.predictions.wait(prediction)
                result = prediction.output
            else:
                # Sync prediction for file input
                result = replicate.run(self.model_version, input=input_params)

            elapsed_time = time.time() - start_time

            if result is None:
                raise RuntimeError("Replicate returned empty result")

            print(f"✅ Transcription completed in {elapsed_time:.2f} seconds")

            # Parse and validate result
            if isinstance(result, dict):
                transcription_result = result
            else:
                # If result is a string, wrap it
                transcription_result = {"text": str(result), "segments": []}

            # Ensure required fields exist
            if "text" not in transcription_result:
                transcription_result["text"] = ""
            if "segments" not in transcription_result:
                transcription_result["segments"] = []

            # Log result summary
            text_length = len(transcription_result.get("text", ""))
            segments_count = len(transcription_result.get("segments", []))
            print(f"📊 Result summary:")
            print(f"   - Text length: {text_length} characters")
            print(f"   - Segments: {segments_count}")

            if segments_count > 0:
                first_segment = transcription_result["segments"][0]
                print(f"   - First segment: {first_segment}")

            return transcription_result

        except Exception as e:
            print(f"❌ Replicate transcription failed: {e}")
            raise RuntimeError(f"Replicate transcription failed: {e}")

    def transcribe_audio_sync(
        self,
        audio_file_path: str,
        language: str = "tr",
        batch_size: int = 24,
        return_timestamps: bool = True,
        enable_diarization: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper for transcribe_audio_async."""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                print("⚠️  Already in event loop, using synchronous Replicate calls")
                # If we're in a loop, we need to use sync version
                return self._transcribe_audio_sync_direct(
                    audio_file_path=audio_file_path,
                    language=language,
                    batch_size=batch_size,
                    return_timestamps=return_timestamps,
                    enable_diarization=enable_diarization
                )
            except RuntimeError:
                # No event loop running, safe to create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run async function
                return loop.run_until_complete(
                    self.transcribe_audio_async(
                        audio_file_path=audio_file_path,
                        language=language,
                        batch_size=batch_size,
                        return_timestamps=return_timestamps,
                        enable_diarization=enable_diarization
                    )
                )
        except Exception as e:
            print(f"❌ Synchronous transcription failed: {e}")
            raise

    def _get_language_name(self, language_code: str) -> str:
        """Convert language code to full language name for Replicate API."""
        language_map = {
            "tr": "turkish",
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "ru": "russian",
            "ja": "japanese",
            "ko": "korean",
            "zh": "chinese",
            "ar": "arabic",
            "hi": "hindi"
        }
        return language_map.get(language_code.lower(), language_code)

    def _transcribe_audio_sync_direct(
        self,
        audio_file_path: str,
        language: str = "tr",
        batch_size: int = 24,
        return_timestamps: bool = True,
        enable_diarization: bool = True
    ) -> Dict[str, Any]:
        """Direct synchronous transcription when already in an event loop."""
        # Initialize client on first use
        self._initialize_client()

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Get HuggingFace token from environment (required for WhisperX diarization)
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HUGGINGFACE_TOKEN not found in environment variables. "
                "Required for WhisperX diarization. Please set it in your .env file."
            )

        # Try to create S3 pre-signed URL first (fastest option)
        s3_url = self._create_s3_presigned_url(audio_file_path)

        # Prepare input parameters for WhisperX model
        input_params = {
            "language": language,  # ISO language code (e.g., "tr" for Turkish)
            "diarization": True,  # Enable speaker diarization with pyannote
            "align_output": True,  # Enable word-level timestamps
            "huggingface_access_token": hf_token,  # Required for pyannote diarization
            "batch_size": 64,  # Default batch size
            "temperature": 0,  # Sampling temperature
            "vad_onset": 0.5,  # Voice activity detection onset
            "vad_offset": 0.363,  # Voice activity detection offset
            "initial_prompt": "",  # Empty initial prompt
            "language_detection_min_prob": 0,  # Language detection probability threshold
            "language_detection_max_tries": 5,  # Max language detection attempts
            "debug": False  # Disable debug output
            # min_speakers: omitted (auto-detect)
            # max_speakers: omitted (auto-detect)
        }

        # Use S3 URL if available, otherwise use file path directly
        if s3_url:
            input_params["audio_file"] = s3_url
            print(f"📡 Using S3 URL for audio input: {os.path.basename(audio_file_path)}")
        else:
            # Use file path directly - Replicate will handle the file reading
            input_params["audio_file"] = open(audio_file_path, "rb")
            print(f"📁 Using direct file upload: {os.path.basename(audio_file_path)}")

        try:
            print(f"🚀 Starting Replicate WhisperX transcription (sync mode)...")
            print(f"   Model: victor-upmeet/whisperx")
            print(f"   Language: {language}")
            print(f"   Diarization: {input_params['diarization']}")
            print(f"   Align output (word-level): {input_params['align_output']}")
            print(f"   Auto-detect speakers: True (min/max omitted)")

            start_time = time.time()

            # Use synchronous replicate.run for direct execution
            result = replicate.run(self.model_version, input=input_params)

            elapsed_time = time.time() - start_time

            # Close file handle if we opened one
            if not s3_url and "audio_file" in input_params:
                try:
                    input_params["audio_file"].close()
                except:
                    pass

            if result is None:
                raise RuntimeError("Replicate returned empty result")

            print(f"✅ Transcription completed in {elapsed_time:.2f} seconds")

            # Parse and validate result
            if isinstance(result, dict):
                transcription_result = result
            else:
                # If result is a string, wrap it
                transcription_result = {"text": str(result), "segments": []}

            # Ensure required fields exist
            if "text" not in transcription_result:
                transcription_result["text"] = ""
            if "segments" not in transcription_result:
                transcription_result["segments"] = []

            # Log result summary
            text_length = len(transcription_result.get("text", ""))
            segments_count = len(transcription_result.get("segments", []))
            print(f"📊 Result summary:")
            print(f"   - Text length: {text_length} characters")
            print(f"   - Segments: {segments_count}")

            if segments_count > 0:
                first_segment = transcription_result["segments"][0]
                print(f"   - First segment: {first_segment}")

            return transcription_result

        except Exception as e:
            print(f"❌ Replicate transcription failed: {e}")
            raise RuntimeError(f"Replicate transcription failed: {e}")


# Global instance (lazy initialization)
replicate_transcription_service = None

def get_replicate_service() -> ReplicateTranscriptionService:
    """Get the global Replicate transcription service instance."""
    global replicate_transcription_service
    if replicate_transcription_service is None:
        replicate_transcription_service = ReplicateTranscriptionService()
    return replicate_transcription_service


def transcribe_with_replicate(
    audio_file_path: str,
    language: str = "tr",
    enable_diarization: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to transcribe audio with Replicate.

    Args:
        audio_file_path: Path to audio file
        language: Language code (default: "tr")
        enable_diarization: Enable speaker diarization

    Returns:
        Transcription result with segments and text
    """
    service = get_replicate_service()
    return service.transcribe_audio_sync(
        audio_file_path=audio_file_path,
        language=language,
        enable_diarization=enable_diarization
    )


async def transcribe_with_replicate_async(
    audio_file_path: str,
    language: str = "tr",
    enable_diarization: bool = True
) -> Dict[str, Any]:
    """
    Async convenience function to transcribe audio with Replicate.

    Args:
        audio_file_path: Path to audio file
        language: Language code (default: "tr")
        enable_diarization: Enable speaker diarization

    Returns:
        Transcription result with segments and text
    """
    service = get_replicate_service()
    return await service.transcribe_audio_async(
        audio_file_path=audio_file_path,
        language=language,
        enable_diarization=enable_diarization
    )