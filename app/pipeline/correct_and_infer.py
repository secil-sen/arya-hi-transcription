import asyncio
import json
import json5
import re
from typing import List, Dict, Any, Optional
from app.pipeline.config import GPT_CORRECTION_AND_NAME_EXTRACTION_PROMPT, JSON_SCHEMA, NEW_USER_PROMPT, NEW_SYSTEM_PROMPT
from app.core.model_registry import models

# -------- Helpers --------
def _parse_gpt_json_lenient(content: str) -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(content)
    except Exception:
        return None

    # 1) Doğrudan liste ise
    if isinstance(data, list):
        return data

    # 2) Obje ise ve "segments" barındırıyorsa
    if isinstance(data, dict):
        if "segments" in data and isinstance(data["segments"], list):
            return data["segments"]
        # 3) Obje ama tek segment döndürülmüşse
        #    (schema object → tek kaydı listeye çevir)
        expected_keys = {"text_corrected", "mentioned_attendees", "multiple_speakers", "candidate_speakers"}
        if expected_keys.issubset(set(data.keys())):
            return [data]

    return None


def _clean_content_for_json(text: str) -> str:
    # Strip code fences like ```json ... ``` or ``` ... ```
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL|re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text.strip()

def _extract_json_block(text: str) -> Optional[str]:
    # Try to find the first JSON array or object in free-form text
    # This is a conservative regex to avoid over-matching
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return m.group(1) if m else None

def _parse_gpt_json(content: str) -> Optional[List[Dict[str, Any]]]:
    if not content or not isinstance(content, str):
        return None

    # 1) Clean code fences/markdown
    cleaned = _clean_content_for_json(content)

    # 2) Try strict JSON
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # 3) If strict failed, try JSON5
        try:
            obj = json5.loads(cleaned)
        except Exception:
            # 4) As a last resort, extract the first JSON-looking block and try again
            candidate = _extract_json_block(cleaned)
            if not candidate:
                return None
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    obj = json5.loads(candidate)
                except Exception:
                    return None

    # Normalize accepted shapes
    if isinstance(obj, dict) and "segments" in obj and isinstance(obj["segments"], list):
        return obj["segments"]
    if isinstance(obj, list):
        return obj

    return None


# -------- Core --------

async def _process_segment_v3(
    i: int,
    segment: Dict[str, Any],
    segments: List[Dict[str, Any]],
    attendee_list: List[str],
    term_list: List[str],
    model: str,
    semaphore: asyncio.Semaphore,
    attendee_id_map: Dict[str, str],
    used_ids: set,
    enable_json_mode: bool = True,
    debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # Process a single segment with retries, JSON-mode, and diagnostics.
    prev_text = segments[i - 1]["text"] if i > 0 else ""
    current_text = segment["text"]
    next_text = segments[i + 1]["text"] if i < len(segments) - 1 else ""

    prev_speaker_id = segments[i - 1]["speaker"] if i > 0 else ""
    current_speaker_id = segment["speaker"]
    next_speaker_id = segments[i + 1]["speaker"] if i < len(segments) - 1 else ""

    prompt = NEW_USER_PROMPT.format(
        term_list=json.dumps(term_list, ensure_ascii=False),
        attendee_list=json.dumps(attendee_list, ensure_ascii=False),
        attendee_id_map=json.dumps(attendee_id_map, ensure_ascii=False),
        current_text=current_text,
        current_speaker_id=current_speaker_id,
        prev_text=prev_text,
        prev_speaker_id=prev_speaker_id,
        next_text=next_text,
        next_speaker_id=next_speaker_id
    )

    max_retries = 3
    last_error: Optional[str] = None

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await models.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": NEW_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
                    # max_tokens=384,  # isterseniz sınırlayın
                )

            # Bazı SDK sürümlerinde content boş dönebilir; yine de JSON string bekliyoruz
            content = ""
            if getattr(response, "choices", None) and response.choices[0].message:
                content = response.choices[0].message.content or ""

            parsed_list = _parse_gpt_json_lenient(content)

            if parsed_list is None:
                # Yalnızca ŞEMA/PARSE hatasında retry
                last_error = "parse_failed"
                if debug_dir:
                    from pathlib import Path
                    Path(debug_dir).mkdir(parents=True, exist_ok=True)
                    Path(f"{debug_dir}/segment_{i}_prompt.txt").write_text(prompt)
                    Path(f"{debug_dir}/segment_{i}_raw.txt").write_text(str(content))
                # Exponential backoff + jitter
                await asyncio.sleep((2 ** attempt) + (0.1 * attempt))
                continue

            # ---------- Normalize ----------
            for item in parsed_list:
                item["start"] = segment.get("start")
                item["end"] = segment.get("end")
                item["speaker"] = segment.get("speaker", "")
                item["source"] = "gpt_split" if len(parsed_list) > 1 else "gpt_single"
                inferred_id = item.get("inferred_speaker_id")
                if inferred_id and inferred_id not in (None, "null"):
                    used_ids.add(inferred_id)

            return parsed_list  # Başarılı parse → hemen çık

        except Exception as e:
            # Yalnızca geçici hatalarda retry mantıklı
            last_error = f"{type(e).__name__}: {e}"
            if debug_dir:
                from pathlib import Path
                Path(debug_dir).mkdir(parents=True, exist_ok=True)
                Path(f"{debug_dir}/segment_{i}_prompt.txt").write_text(prompt)
                Path(f"{debug_dir}/segment_{i}_exception.txt").write_text(last_error or "")
            await asyncio.sleep((2 ** attempt) + (0.1 * attempt))

    # ---------- Tüm denemeler başarısız ----------
    segment_copy = {
        "text_corrected": segment.get("text", ""),
        "mentioned_attendees": [],
        "multiple_speakers": False,
        "inferred_speaker": "None",
        "inferred_speaker_id": None,
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker": segment.get("speaker", ""),
        "source": "error",
    }
    if debug_dir and last_error:
        segment_copy["error_detail"] = last_error
    return [segment_copy]


async def gpt_correct_and_infer_segments_async_v3(
    segments: List[Dict[str, Any]],
    attendee_list: List[str],
    term_list: List[str],
    attendee_id_map: Dict[str, str],
    model: str = "gpt-4o-mini",
    concurrency_limit: int = 2,
    enable_json_mode: bool = True,
    debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Runs correction + speaker inference asynchronously.
    - Stops speaker inference early if all attendee IDs are already inferred.
    - Uses JSON mode and object wrapper to ensure parseable output.
    - Writes prompt/raw/exception diagnostics when parsing fails (if debug_dir provided).
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    used_ids: set = set()
    all_ids = set(attendee_id_map.values())

    results: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        if used_ids == all_ids:
            seg_out = {
                "text_corrected": seg.get("text", ""),
                "mentioned_attendees": [],
                "multiple_speakers": False,
                "inferred_speaker": "None",
                "inferred_speaker_id": None,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "speaker": seg.get("speaker", ""),
                "source": "skipped_after_infer_complete",
            }
            results.append(seg_out)
            continue

        parsed = await _process_segment_v3(
            i=i,
            segment=seg,
            segments=segments,
            attendee_list=attendee_list,
            term_list=term_list,
            model=model,
            semaphore=semaphore,
            attendee_id_map=attendee_id_map,
            used_ids=used_ids,
            enable_json_mode=enable_json_mode,
            debug_dir=debug_dir,
        )
        results.extend(parsed)

    return results