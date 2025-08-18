from pydantic import BaseModel
from typing import List, Optional

class TranscribeRequest(BaseModel):
    user_id: Optional[str] = "anonymous"
    video_path:str
    attendees:List[str]

class TranscribeResponse(BaseModel):
    json_output:dict

