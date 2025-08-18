from fastapi import FastAPI, HTTPException
from app.schemas import TranscribeRequest, TranscribeResponse
from app.pipeline.pipeline_old import run_transcript_pipeline

app = FastAPI(title="Arya HI Transcript Extraction API")

@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_video(request:TranscribeRequest):
    try:
        result = run_transcript_pipeline(mp4_path=request.video_path,
                                         user_id=request.user_id,
                                         attendees=request.attendees)
        return {"json_output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))