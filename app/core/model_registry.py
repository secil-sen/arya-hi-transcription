import whisper
from pyannote.audio import Pipeline
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pyannote.audio import Inference
from app.core.config import (WHISPER_MODEL_NAME,
                             DIARIZATION_MODEL_NAME,
                             EMBEDDING_MODEL_NAME)


class ModelRegistry:
    def __init__(self):
        print("Loading models eagerly...")
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            raise EnvironmentError("HUGGINGFACE_TOKEN not set.")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise EnvironmentError("HUGGINGFACE_TOKEN not set.")

        self.whisper = whisper.load_model(WHISPER_MODEL_NAME)
        self.diarization = Pipeline.from_pretrained(DIARIZATION_MODEL_NAME,
                                                    use_auth_token=hf_token)
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)  # Veya os.getenv("OPENAI_API_KEY")
        self.inference = Inference(EMBEDDING_MODEL_NAME, window="whole")

load_dotenv()
models = ModelRegistry() # Global singleton instance.

