# Arya HI Transcript Extraction API

A video transcription service that uses Google Gemini for text correction and speaker inference.

## Setup Instructions

### 1. Clone and enter the project
```bash
git clone <your-private-repo-url> arya-hi-transcription
cd arya-hi-transcription
```

### 2. Point Poetry to Python 3.11 and install dependencies
```bash
poetry env use python3.11
# or the full path, e.g. /usr/local/bin/python3.11
poetry install
```

### 3. Create your local environment file
```bash
cp .env.example .env
# Edit .env and fill the required keys.
```

Required environment variables:
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `HUGGINGFACE_TOKEN`: Your Hugging Face token for audio models
- `DEBUG_MODE`: Set to "true" for debug mode (optional)
- `OUTPUT_ROOT`: Output directory path (optional)

### 4. Run the API
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

• Health check: http://127.0.0.1:8000/health

## API Endpoints

- `POST /transcribe`: Transcribe a video file
  - Request body: `{"user_id": "string", "video_path": "string", "attendees": ["string"]}`
  - Response: `{"json_output": {...}}`

## Features

- Video to audio conversion using FFmpeg
- Audio chunking for processing
- Speaker diarization using Pyannote
- Speech-to-text transcription using Whisper
- Text correction and speaker inference using Google Gemini
- JSON output with speaker identification

## Dependencies

- Python 3.11+
- FastAPI for web framework
- Google Generative AI for text processing
- Whisper for speech recognition
- Pyannote for speaker diarization
- Poetry for dependency management
