from pydantic import BaseModel
from typing import List, Optional

class TranscribeRequest(BaseModel):
    user_id: Optional[str] = "anonymous"
    video_path: str
    attendees: List[str]
    meeting_id: str

class ContextChunk(BaseModel):
    text: str
    speaker_name: str
    start_ms: int
    end_ms: int
    meeting_id: str
    score: float

class AutoAnalysisData(BaseModel):
    prompt: str
    answer: str
    context_chunks: List[ContextChunk]
    search_type: Optional[str] = None
    processing_time_ms: Optional[int] = None
    chunk_count: Optional[int] = None

class AutoAnalysisNotification(BaseModel):
    type: str
    notification_id: Optional[str] = None
    org_id: Optional[str] = None
    meeting_id: str
    auto_analysis: AutoAnalysisData
    generated_at: Optional[str] = None

class TranscribeResponse(BaseModel):
    json_output: dict

