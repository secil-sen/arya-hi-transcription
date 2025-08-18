import json
import os
from typing import List, Dict

def save_as_json(data, output_path:str, ensure_dir:bool=True) -> None:
    if ensure_dir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False) #ensure_Ascii=False for protecting Turkish special chars.

def filter_short_segments(segments: List[Dict], min_duration: float = 1.0) -> List[Dict]:
    filtered = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        if duration >= min_duration or len(seg["text"].strip()) > 15:
            filtered.append(seg)
    return filtered


