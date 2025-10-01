import httpx
import hashlib
import json
import os
from typing import Optional


def notify_chunking(meeting_id: str, transcript_path: str, chunking_url: str = "http://localhost:8090") -> dict:
    """
    Send notification to chunking service when transcript is ready.
    
    Args:
        meeting_id: The meeting ID associated with the transcript
        transcript_path: Path to the transcript file (JSONL format)
        chunking_url: URL of the chunking service (default: localhost:8090)
    
    Returns:
        Response from chunking service
    """
    try:
        # Count lines in transcript file
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        count = len(lines)

        # Calculate SHA256 checksum
        sha256 = hashlib.sha256()
        with open(transcript_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        checksum = sha256.hexdigest()

        # Prepare payload
        payload = {
            "type": "finalize_ready",
            "meeting_id": meeting_id,
            "object_uri": f"file://{os.path.abspath(transcript_path)}",
            "version": 1,
            "count": count,
            "checksum": checksum,
        }

        # Send notification
        url = f"{chunking_url}/meetings/notify"
        print(f"Sending notification to chunking service: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
            
    except Exception as e:
        print(f"Failed to notify chunking service: {e}")
        raise RuntimeError(f"Notification failed: {e}")


def save_as_jsonl(data: list, output_path: str, ensure_dir: bool = True) -> None:
    """
    Save data as JSONL (JSON Lines) format where each line is a separate JSON object.
    
    Args:
        data: List of dictionaries to save
        output_path: Path where to save the file
        ensure_dir: Whether to create parent directories if they don't exist
    """
    if ensure_dir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
