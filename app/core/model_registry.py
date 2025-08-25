import whisper
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pyannote.audio import Inference
from app.core.config import (WHISPER_MODEL_NAME,
                             DIARIZATION_MODEL_NAME,
                             EMBEDDING_MODEL_NAME)


class ModelRegistry:
    def __init__(self):
        self._whisper = None
        self._diarization = None
        self._gemini_model = None
        self._inference = None
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
            self._ensure_initialized()
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            self._diarization = Pipeline.from_pretrained(DIARIZATION_MODEL_NAME,
                                                        use_auth_token=hf_token)
        return self._diarization

    @property
    def gemini_model(self):
        if self._gemini_model is None:
            self._ensure_initialized()
        return self._gemini_model

    @property
    def inference(self):
        if self._inference is None:
            self._ensure_initialized()
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            self._inference = Inference(EMBEDDING_MODEL_NAME, 
                                      use_auth_token=hf_token,
                                      window="whole")
        return self._inference

load_dotenv()
models = ModelRegistry() # Global singleton instance.

