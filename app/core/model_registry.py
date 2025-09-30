import whisper
try:
    from pyannote.audio import Pipeline
except ImportError:
    try:
        from pyannote.pipeline import Pipeline
    except ImportError:
        Pipeline = None
import os
from dotenv import load_dotenv
import google.generativeai as genai
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
from app.core.config import (WHISPER_MODEL_NAME,
                             DIARIZATION_MODEL_NAME,
                             EMBEDDING_MODEL_NAME)

# Import Replicate transcription service (lazy loading)
try:
    from app.pipeline.replicate_transcription import get_replicate_service
    REPLICATE_AVAILABLE = True
    print("✅ Replicate transcription service available")
except ImportError as e:
    REPLICATE_AVAILABLE = False
    get_replicate_service = None
    print(f"⚠️  Replicate transcription service not available: {e}")


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

            # Check if using Vertex AI or Google AI
            use_vertex_ai = os.getenv("USE_VERTEX_AI", "false").lower() == "true"

            if use_vertex_ai and VERTEX_AI_AVAILABLE:
                # Configure Vertex AI
                project_id = os.getenv("VERTEX_AI_PROJECT_ID")
                location = os.getenv("VERTEX_AI_LOCATION", "us-central1")

                if not project_id:
                    raise EnvironmentError("VERTEX_AI_PROJECT_ID not set for Vertex AI usage.")

                vertexai.init(project=project_id, location=location)
                # Use correct Vertex AI model path format
                self._gemini_model = GenerativeModel("publishers/google/models/gemini-2.5-flash")

                # Set generation config for Vertex AI
                self._generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            else:
                # Configure Google AI (default)
                genai.configure(api_key=gemini_api_key)
                self._gemini_model = genai.GenerativeModel('gemini-1.5-flash')

                # Set generation config for Google AI
                self._gemini_model.generation_config = genai.types.GenerationConfig(
                    temperature=0.1,
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
            print("Checking diarization model...")

            # Check if Pipeline is available
            if Pipeline is None:
                print("WARNING: pyannote.audio Pipeline not available - diarization will be skipped")
                print("Please install pyannote.audio>=3.1.1:")
                print("  pip install 'pyannote.audio>=3.1.1'")
                return None

            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                print("WARNING: HUGGINGFACE_TOKEN not found - diarization will be skipped")
                return None

            try:
                print(f"Loading diarization model: {DIARIZATION_MODEL_NAME}")

                # Try new API first (pyannote.audio >= 3.0)
                if hasattr(Pipeline, 'from_pretrained'):
                    self._diarization = Pipeline.from_pretrained(
                        DIARIZATION_MODEL_NAME,
                        use_auth_token=hf_token
                    )
                else:
                    print("WARNING: pyannote.audio version doesn't support Pipeline.from_pretrained")
                    print("Please upgrade to pyannote.audio>=3.1.1:")
                    print("  pip install --upgrade 'pyannote.audio>=3.1.1'")
                    self._diarization = None
                    return None

                print("Diarization model loaded successfully")
            except Exception as e:
                print(f"WARNING: Failed to load diarization model: {e}")
                print("This might be due to:")
                print("  - Missing HuggingFace token")
                print("  - Network connectivity issues")
                print("  - Model access permissions (make sure you accepted the model license)")
                print("  - Incompatible pyannote.audio version")
                print("Returning None - diarization will be skipped")
                self._diarization = None
        return self._diarization

    @property
    def gemini_model(self):
        if self._gemini_model is None:
            self._ensure_initialized()
        return self._gemini_model

    @property
    def inference(self):
        if getattr(self, "_inference", None) is None:
            self._ensure_initialized()
            print("Checking inference model...")
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                print("WARNING: HUGGINGFACE_TOKEN not found - inference will be skipped")
                return None
            try:
                try:
                    from pyannote.audio import Inference
                    from pyannote.audio import Model
                except ImportError:
                    print("WARNING: pyannote.audio Inference/Model not available - using diarization pipeline's internal embedding extraction")
                    self._inference = None
                    return None

                print(f"Loading inference model: {EMBEDDING_MODEL_NAME}")

                # Try to load using Inference class first (preferred for pyannote.audio 3.x)
                try:
                    self._inference = Inference(
                        EMBEDDING_MODEL_NAME,
                        use_auth_token=hf_token
                    )
                    print("Inference model loaded successfully using Inference class")
                except Exception as inference_error:
                    print(f"Failed to load with Inference class: {inference_error}")
                    # Fallback to Model class
                    try:
                        if hasattr(Model, 'from_pretrained'):
                            self._inference = Model.from_pretrained(
                                EMBEDDING_MODEL_NAME,
                                use_auth_token=hf_token
                            )
                            print("Inference model loaded successfully using Model class")
                        else:
                            print("WARNING: pyannote.audio Model.from_pretrained not available")
                            print("Using diarization pipeline's internal embedding extraction")
                            self._inference = None
                    except Exception as model_error:
                        print(f"Failed to load with Model class: {model_error}")
                        print("Using diarization pipeline's internal embedding extraction")
                        self._inference = None
            except Exception as e:
                print(f"WARNING: Failed to load inference model: {e}")
                print("This might be due to:")
                print("  - Missing HuggingFace token")
                print("  - Network connectivity issues")
                print("  - Model access permissions (make sure you accepted the model license)")
                print("  - Incompatible pyannote.audio version")
                print("Using diarization pipeline's internal embedding extraction")
                self._inference = None
        return self._inference

    @property
    def replicate_transcription(self):
        """Access Replicate transcription service."""
        if REPLICATE_AVAILABLE and get_replicate_service is not None:
            return get_replicate_service()
        else:
            print("WARNING: Replicate transcription service not available")
            return None

load_dotenv()
models = ModelRegistry() # Global singleton instance.

