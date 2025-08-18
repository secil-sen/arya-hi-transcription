import whisper
from pyannote.audio import Pipeline
import os

class LazyModelRegistry:
    _whisper = None
    _diarization = None

    @classmethod
    def get_whisper(cls):
        if cls._whisper is None:
            print("Loading whisper model...")
            cls._whisper = whisper.load_model("large")
        return cls._whisper

    @classmethod
    def get_diarization(cls):
        if cls._diarization is None:
            print("Loading diarization model...")
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise EnvironmentError("HUGGINGFACE_TOKEN not found.")
            cls._diarization = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=hf_token
            )
        return cls._diarization