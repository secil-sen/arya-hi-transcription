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
- `CHUNKING_SERVICE_URL`: URL of the chunking service for notifications (default: http://localhost:8090)

### 4. Run the API
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

• Health check: http://127.0.0.1:8000/health

## API Endpoints

- `POST /transcribe`: Transcribe a video file
  - Request body: 
    ```json
    {
      "user_id": "string",
      "video_path": "string", 
      "attendees": ["string"],
      "meeting_id": "string"
    }
    ```
  - Response: `{"json_output": {...}}`

- `GET /health`: Health check endpoint

## Features

- Video to audio conversion using FFmpeg
- Audio chunking for processing
- Speaker diarization using Pyannote
- Speech-to-text transcription using Whisper
- Text correction and speaker inference using Google Gemini
- JSON and JSONL output with speaker identification
- Automatic notification to chunking service when transcript is ready
- SHA256 checksum validation for transcript files

## Notification System

The service automatically sends notifications to a chunking service when transcript processing is complete. The notification includes:

- `type`: "finalize_ready" 
- `meeting_id`: The meeting identifier
- `object_uri`: File path to the transcript (JSONL format)
- `version`: Transcript version (currently 1)
- `count`: Number of transcript lines
- `checksum`: SHA256 hash of the transcript file

The notification is sent to `{CHUNKING_SERVICE_URL}/meetings/notify` endpoint.

## Testing

Run the test script to verify the notification functionality:

```bash
python test_notification.py
```

## Dependencies

- Python 3.11+
- FastAPI for web framework
- Google Generative AI for text processing
- Whisper for speech recognition
- Pyannote for speaker diarization
- httpx for HTTP notifications
- Poetry for dependency management
