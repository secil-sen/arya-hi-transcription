import asyncio
import json
import json5
import re
from typing import List, Dict, Any, Optional
from app.pipeline.config import JSON_SCHEMA
from app.core.model_registry import models

# -------- Helpers --------
def _parse_gemini_json_lenient(content: str) -> Optional[List[Dict[str, Any]]]:
    if not content or not isinstance(content, str):
        return None
    
    # Clean the content first
    cleaned = _clean_content_for_json(content)
    
    # Try multiple parsing strategies
    parsed_data = None
    
    # Strategy 1: Direct JSON parsing
    try:
        parsed_data = json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Try JSON5 if direct parsing failed
    if parsed_data is None:
        try:
            parsed_data = json5.loads(cleaned)
        except Exception:
            pass
    
    # Strategy 3: Extract JSON block if still failed
    if parsed_data is None:
        extracted = _extract_json_block(cleaned)
        if extracted:
            try:
                parsed_data = json.loads(extracted)
            except json.JSONDecodeError:
                try:
                    parsed_data = json5.loads(extracted)
                except Exception:
                    pass
    
    # Strategy 4: Try to find and parse just the JSON part
    if parsed_data is None:
        # Look for content between curly braces
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, cleaned)
        if matches:
            for match in matches:
                try:
                    parsed_data = json.loads(match)
                    break
                except json.JSONDecodeError:
                    continue
    
    # Strategy 5: Try to find the first valid JSON object in the text
    if parsed_data is None:
        import re
        # Look for the first { and last } to extract JSON
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_candidate = cleaned[start:end+1]
            try:
                parsed_data = json.loads(json_candidate)
            except json.JSONDecodeError:
                try:
                    parsed_data = json5.loads(json_candidate)
                except Exception:
                    pass
    
    if parsed_data is None:
        return None

    # Normalize the parsed data
    if isinstance(parsed_data, list):
        return parsed_data
    elif isinstance(parsed_data, dict):
        # If it's a dict with segments, return segments
        if "segments" in parsed_data and isinstance(parsed_data["segments"], list):
            return parsed_data["segments"]
        # If it's a single segment dict, return as list
        elif any(key in parsed_data for key in ["text_corrected", "inferred_speaker"]):
            return [parsed_data]
    
    return None


def _clean_content_for_json(text: str) -> str:
    # Strip code fences like ```json ... ``` or ``` ... ```
    import re
    
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove any remaining markdown formatting
    text = re.sub(r'^\s*```\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    
    return text

def _extract_json_block(text: str) -> Optional[str]:
    # Try to find the first JSON array or object in free-form text
    # This is a conservative regex to avoid over-matching
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    return m.group(1) if m else None

def _parse_gemini_json(content: str) -> Optional[List[Dict[str, Any]]]:
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

    # Build prompt with context
    prompt = f"""You are a transcription correction and speaker inference assistant. 

Context: 
- Previous segment: "{prev_text}"
- Current segment: "{current_text}"
- Next segment: "{next_text}"

Attendees: {', '.join(attendee_list)}
Term list: {', '.join(term_list)}

Please correct the current segment text and infer the speaker. You MUST return ONLY valid JSON in this exact format:

{{
    "text_corrected": "corrected text here",
    "mentioned_attendees": ["list of mentioned attendees"],
    "multiple_speakers": false,
    "inferred_speaker": "speaker name",
    "inferred_speaker_id": "speaker id from attendee list"
}}

CRITICAL REQUIREMENTS: 
- Return ONLY the raw JSON object, no markdown, no code blocks, no explanations
- Do NOT wrap in ```json or ``` blocks
- Do NOT add any text before or after the JSON
- Ensure the JSON is properly formatted and valid
- If you cannot determine a speaker, use "Unknown" for inferred_speaker and null for inferred_speaker_id
- mentioned_attendees should be an empty array if no attendees are mentioned
- Do NOT add ellipsis ("...") between words or sequences, even if they seem short or incomplete
- Preserve the original text structure without adding punctuation or filler characters

Example of what NOT to do:
❌ ```json
❌ {{
❌ }}

Example of what TO do:
✅ {{
✅ }}"""

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            async with semaphore:
                # Use Gemini instead of OpenAI
                response = await asyncio.to_thread(
                    models.gemini_model.generate_content,
                    prompt
                )

            # Extract content from Gemini response
            content = response.text if hasattr(response, 'text') else str(response)
            
            # Debug logging
            print(f"Segment {i} - Raw Gemini response: {content[:200]}...")
            print(f"Segment {i} - Response type: {type(response)}")
            print(f"Segment {i} - Has text attr: {hasattr(response, 'text')}")
            
            # Safety check for empty or invalid content
            if not content or content.strip() == "":
                print(f"Segment {i} - Empty response from Gemini")
                last_error = "empty_response"
                await asyncio.sleep((2 ** attempt) + (0.1 * attempt))
                continue

            parsed_list = _parse_gemini_json_lenient(content)
            
            print(f"Segment {i} - Parsed result: {parsed_list}")

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
                item["source"] = "gemini_split" if len(parsed_list) > 1 else "gemini_single"
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
    print(f"Segment {i} - All parsing attempts failed. Using fallback.")
    
    # Create a fallback segment with the original text
    segment_copy = {
        "text_corrected": segment.get("text", ""),
        "mentioned_attendees": [],
        "multiple_speakers": False,
        "inferred_speaker": "Unknown",
        "inferred_speaker_id": None,
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker": segment.get("speaker", ""),
        "source": "fallback_after_parse_failure",
        "error_detail": last_error or "JSON parsing failed after all retries"
    }
    
    if debug_dir and last_error:
        from pathlib import Path
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{debug_dir}/segment_{i}_fallback.txt").write_text(f"Fallback used. Error: {last_error}")
    
    return [segment_copy]


async def gemini_correct_and_infer_segments_async_v3(
    segments: List[Dict[str, Any]],
    attendee_list: List[str],
    term_list: List[str],
    attendee_id_map: Dict[str, str],
    model: str = "gemini-1.5-flash",
    concurrency_limit: int = 2,
    enable_json_mode: bool = True,
    debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Runs correction + speaker inference asynchronously using Google Gemini.
    - Stops speaker inference early if all attendee IDs are already inferred.
    - Uses structured prompts to ensure parseable output.
    - Writes prompt/raw/exception diagnostics when parsing fails (if debug_dir provided).
    """
    
    # Safety check for empty segments
    if not segments:
        print("WARNING: No segments provided to Gemini processing!")
        return []
    
    print(f"Processing {len(segments)} segments with Gemini...")
    print(f"First segment sample: {segments[0] if segments else 'None'}")
    
    # Validate segment structure
    required_fields = ["text", "start", "end", "speaker"]
    for i, seg in enumerate(segments):
        missing_fields = [field for field in required_fields if field not in seg]
        if missing_fields:
            print(f"WARNING: Segment {i} missing required fields: {missing_fields}")
            print(f"Segment data: {seg}")
    
    semaphore = asyncio.Semaphore(concurrency_limit)
    used_ids: set = set()
    all_ids = set(attendee_id_map.values())

    results: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        print(f"Processing segment {i}/{len(segments)}: {seg.get('text', '')[:50]}...")
        
        if used_ids == all_ids:
            print(f"All attendee IDs already inferred, skipping segment {i}")
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
        
        print(f"Segment {i} processed, got {len(parsed)} results")
        results.extend(parsed)

    print(f"Gemini processing complete. Total results: {len(results)}")
    if results:
        print(f"Sample result: {results[0]}")
    
    return results


# -------- Name Correction with Gemini --------

NAME_CORRECTION_PROMPT = """
Sen bir isim düzeltme asistanısın. Sana verilen transkript segmentinde geçen isimleri analiz et ve hataları düzelt.

**Kurallar:**
1. Marka isimleri büyük harfle yazılmalı (ARYA, IBM, AWS gibi)
2. Türkçe karakter hataları düzeltilmeli (gürkem → Görkem, şükrü → Şükrü)
3. İngilizce isimler düzgün kapitalize edilmeli (john → John, mary → Mary)
4. Yaygın yazım hataları düzeltilmeli (aria → ARYA, arial → ARYA, jon → John)
5. Baş harfleri büyük olmalı (örn: görkem çetin → Görkem Çetin)

**Input Segment:**
{segment_text}

**Çıkarılan İsimler:**
{extracted_names}

**Beklenen Çıktı (JSON):**
{{
    "corrected_names": [
        {{"original": "aria", "corrected": "ARYA", "type": "brand"}},
        {{"original": "gürkem", "corrected": "Görkem", "type": "person"}}
    ]
}}

Sadece JSON formatında cevap ver, açıklama ekleme. ÖNEMLI: Eğer tüm isimler zaten doğruysa, boş bir liste dön: {{"corrected_names": []}}
"""


async def enrich_names_with_gemini(segments: List[Dict]) -> List[Dict]:
    """
    NER ile çıkarılan isimleri Gemini'ye gönderip hataları düzeltir.

    Örnek:
    - "Aria" → "ARYA" (marka ismi)
    - "gürkem" → "Görkem" (türkçe karakter)
    - "jon doe" → "John Doe" (İngilizce isim)

    Args:
        segments: NER extraction sonrası segment listesi (extracted_names field ile)

    Returns:
        Düzeltilmiş isimlerle güncellenmiş segment listesi
    """

    import time
    from app.core.model_registry import models

    start_time = time.time()
    corrections_count = 0
    correction_examples = []

    print("Starting name correction with Gemini...")

    for idx, segment in enumerate(segments):
        # NER'den çıkan isimler varsa
        if segment.get('extracted_names') and len(segment['extracted_names']) > 0:

            segment_text = segment.get('text_corrected', segment.get('text', ''))
            extracted_names = segment['extracted_names']

            prompt = NAME_CORRECTION_PROMPT.format(
                segment_text=segment_text,
                extracted_names=', '.join(extracted_names)
            )

            try:
                # Gemini'ye gönder
                response = await asyncio.to_thread(
                    models.gemini_model.generate_content,
                    prompt
                )

                # Extract content from Gemini response
                content = response.text if hasattr(response, 'text') else str(response)

                # Debug logging
                print(f"Segment {idx} - Name correction raw response: {content[:200]}...")

                # Safety check for empty response
                if not content or content.strip() == "":
                    print(f"⚠️  Segment {idx}: Empty response from Gemini for name correction")
                    continue

                # Clean and parse JSON response
                cleaned_content = _clean_content_for_json(content)

                # Try to parse as JSON directly
                corrected_data = None
                try:
                    corrected_data = json.loads(cleaned_content)
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    try:
                        json_match = re.search(r'\{[\s\S]*\}', cleaned_content)
                        if json_match:
                            corrected_data = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

                if corrected_data is None:
                    print(f"⚠️  Segment {idx}: Failed to parse Gemini response for name correction")
                    print(f"   Raw content: {content[:300]}")
                    continue

                # Handle corrected names
                corrected_names_list = corrected_data.get('corrected_names', [])

                # If corrected_names is empty, no corrections needed
                if not corrected_names_list or len(corrected_names_list) == 0:
                    print(f"   Segment {idx}: No corrections needed")
                    continue

                # Create mapping from original to corrected
                correction_map = {}
                for item in corrected_names_list:
                    if isinstance(item, dict) and 'original' in item and 'corrected' in item:
                        correction_map[item['original']] = item['corrected']

                # Update extracted_names with corrections
                updated_names = []
                for name in extracted_names:
                    corrected_name = correction_map.get(name, name)
                    updated_names.append(corrected_name)

                    # Track corrections for logging
                    if corrected_name != name:
                        corrections_count += 1
                        if len(correction_examples) < 5:  # Keep first 5 examples
                            correction_examples.append(f"{name} → {corrected_name}")

                # Update segment with corrected names
                segment['extracted_names'] = updated_names
                segment['corrected_names_metadata'] = corrected_names_list

                print(f"   Segment {idx}: Corrected {len(correction_map)} names")

            except Exception as e:
                print(f"⚠️  Segment {idx}: Name correction failed - {e}")
                import traceback
                traceback.print_exc()
                # Continue with original names
                continue

    elapsed_time = time.time() - start_time

    # Logging
    print(f"✅ Name correction completed in {elapsed_time:.2f}s")
    print(f"   - {corrections_count} names corrected")
    if correction_examples:
        print(f"   - Corrections: {', '.join(correction_examples)}")

    return segments