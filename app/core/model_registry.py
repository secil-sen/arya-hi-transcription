import whisper
from pyannote.pipeline import Pipeline
import os
from dotenv import load_dotenv
import google.generativeai as genai
from app.core.config import (WHISPER_MODEL_NAME,
                             DIARIZATION_MODEL_NAME,
                             EMBEDDING_MODEL_NAME)


class ModelRegistry:
    def __init__(self):
        self._whisper = None
        self._diarization = None
        self._gemini_model = None
        # self._inference = None  # Commented out - not available in older version
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            print("Loading models eagerly...")
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise EnvironmentError("HUGGINGFACE_TOKEN not set.")
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise EnvironmentError("GOOGLE_API_KEY not set.")

            # Configure Google Gemini
            genai.configure(api_key=gemini_api_key)
            self._gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Set generation config for better JSON output
            self._gemini_model.generation_config = genai.types.GenerationConfig(
                temperature=0.1,  # Lower temperature for more consistent output
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048,
            )
            self._initialized = True

    @property
    def whisper(self):
        if self._whisper is None:
            self._ensure_initialized()
            self._whisper = whisper.load_model(WHISPER_MODEL_NAME)
        return self._whisper

    @property
    def diarization(self):
        if self._diarization is None:
            print("WARNING: pyannote.audio 1.1.2 doesn't support Pipeline.from_pretrained")
            print("Returning None - diarization will be skipped")
            # In the older version, we would need to configure the pipeline manually
            # with model files, which is complex. For now, we'll disable diarization
            self._diarization = None
        return self._diarization

    @property
    def gemini_model(self):
        if self._gemini_model is None:
            self._ensure_initialized()
        return self._gemini_model

    @property
    def inference(self):
        # Note: Inference not available in pyannote.audio 1.1.2
        # This is a placeholder to prevent import errors
        # The diarization pipeline should handle embeddings internally
        if getattr(self, "_inference", None) is None:
            print("WARNING: Inference model not available in pyannote.audio 1.1.2")
            print("Using diarization pipeline's internal embedding extraction")
            self._inference = None
        return self._inference

load_dotenv()
models = ModelRegistry() # Global singleton instance.

