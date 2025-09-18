# -*- coding: utf-8 -*-
"""
Rule Based Name Resolution

Speaker Name Resolution Pipeline (Windowed + Speaker Canonical Assignment)

This module provides a two-pass pipeline for assigning human-readable participant
names to diarized speaker segments in meeting transcripts. It is designed to work
with transcripts in JSON array format, where each item
represents a diarized and transcribed segment.

## Pass 1: Enrichment
- Extract lexical events (self-identification, addresssee, attribution,
  strong address forms such as "X Bey/Hanım" or "Hello X").
- Optionally run a transformer-based NER model to extract additional person names.
- Collect candidate names from events, mentioned attendees, and candidate_speakers.
- Normalize timestamps to milliseconds and attach enriched metadata.

## Pass 1.5: Speaker Canonical Map
- Aggregate strong signals per speaker ID (e.g., self-identification,
  diarization-continuity, strong NER matches).
- Select a canonical name for each speaker if confidence thresholds and margins
  are satisfied.
- Produce a summary of detected participants, their canonical names, confidence
  scores, and total speaking duration.

## Pass 2: Assignment
- For each segment, assign a name using the speaker canonical map if available
  and no strong conflict is detected.
- If a conflict exists (e.g., self-identification or strong addresssee contradicts
  the canonical map), fall back to a windowed scoring mechanism that considers:
  - Current, previous, and next segment evidence (with exponential decay).
  - Continuity with the previously assigned name.
  - Continuity within the same diarization speaker ID.
- Store top candidate scores and assignment reasons for interpretability.

## Outputs
- The enriched transcript JSON array with added fields:
  - `events`, `ner_names`, `names_union`
  - `assigned_name`, `assigned_confidence`, `assigned_reason`
  - `name_scores` (top scoring candidates)
  - `start_ms`, `end_ms`
- Metadata summary (`_meta_assignment_summary`) including participant count,
  detected speakers, `speaker_id_map`, and `participants_summary`.

## Intended Use
- As part of a larger meeting transcription and retrieval pipeline (see Arya HI
  project plan), this module improves the reliability of speaker name attribution
  when diarization IDs need to be mapped to actual participant names.
- Supports bilingual detection (Turkish and English) and extensible rule-based
  patterns for honorifics, greetings, and attribution.
- Enables downstream formatting and RAG indexing with correct speaker attribution.

## Dependencies
- `rapidfuzz` (for fuzzy matching against a roster list).
- `transformers` + `torch` (optional, for NER if enabled).
"""

import argparse, json, math, re
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
from rapidfuzz import process, fuzz

# Helpers & Regex rules
RE_SPACES = re.compile(r"\s+")
CAP_NAME = r"[A-ZÇĞİÖŞÜ][\wçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][\wçğıöşü]+)?"
def norm_text(s: str) -> str:
    return RE_SPACES.sub(" ", s or "").strip()

PATTERNS = {
    "self_id": [
        re.compile(rf"\bben(?:im)?\s+ad(?:[ıi]m)?\s+({CAP_NAME})", re.IGNORECASE),
        re.compile(rf"\bben\s+({CAP_NAME})\b", re.IGNORECASE),  # "Ben Ahmet"
        re.compile(rf"\bi am\s+({CAP_NAME})\b", re.IGNORECASE),
        re.compile(rf"\bthis is\s+({CAP_NAME})\b", re.IGNORECASE),
        re.compile(rf"\b({CAP_NAME})\s+ben\b", re.IGNORECASE),  # "Ahmet ben"
    ],
    "addresssee": [
        re.compile(rf"\b({CAP_NAME})\s*,?\s*sen\b", re.IGNORECASE),
        re.compile(rf"\bthanks?,?\s+({CAP_NAME})\b", re.IGNORECASE),
        re.compile(rf"\bteşekkürler?,?\s+({CAP_NAME})(?:\s+Hanım|Bey)?\b", re.IGNORECASE),
        re.compile(rf"\b({CAP_NAME})\s*,?\s*could you\b", re.IGNORECASE),
    ],
}
def extract_addresssee_strong(t: str) -> List[str]:
    names = set()
    for m in re.finditer(rf"\b({CAP_NAME})\s*(Bey|Hanım)\b", t, flags=re.IGNORECASE):
        names.add(m.group(1))
    for m in re.finditer(rf"\b(merhaba|selam(?:lar)?)\s+({CAP_NAME})\b", t, flags=re.IGNORECASE):
        names.add(m.group(2))
    for m in re.finditer(rf"\b({CAP_NAME})\s*(?:Bey|Hanım)?\s*,?\s*merhaba\b", t, flags=re.IGNORECASE):
        names.add(m.group(1))
    for m in re.finditer(rf"\bsayın\s+({CAP_NAME})\b", t, flags=re.IGNORECASE):
        names.add(m.group(1))
    return list(names)

def extract_events(text: str) -> Dict[str, List[str]]:
    t = norm_text(text)
    out = {"self_id": [], "addresssee": [], "attribution": [], "addresssee_strong": []}
    for key, regs in PATTERNS.items():
        for r in regs:
            for m in r.finditer(t):
                name = m.group(1)
                if name and name not in out[key]:
                    out[key].append(name)
    for r in [
        re.compile(rf"\b({CAP_NAME})\'?(?:n(?:in|ın|un|ün)|ın|in|un|ün)\s+(dedi[gğ]i|raporu|notu|önerisi)\b", re.IGNORECASE),
        re.compile(rf"\bas\s+({CAP_NAME})\s+mentioned\b", re.IGNORECASE),
        re.compile(rf"\baccording to\s+({CAP_NAME})\b", re.IGNORECASE),
    ]:
        for m in r.finditer(t):
            name = m.group(1)
            if name and name not in out["attribution"]:
                out["attribution"].append(name)
    out["addresssee_strong"] = extract_addresssee_strong(t)
    return out

NER_PIPE = None
def ner_persons(text: str, model_name: Optional[str], thr: float = 0.60) -> List[str]:
    global NER_PIPE
    if not text or not model_name or not model_name.strip():
        return []
    
    # Safety: Limit text length to prevent memory issues
    if len(text) > 1000:
        print(f"Warning: Text too long for NER ({len(text)} chars), truncating to 1000")
        text = text[:1000]
    
    try:
        if NER_PIPE is None:
            print(f"Loading NER model: {model_name} (this may take time and memory...)")
            from transformers import pipeline
            import torch
            
            # Use CPU and optimize for memory
            device = 0 if torch.cuda.is_available() else -1
            NER_PIPE = pipeline(
                "token-classification", 
                model=model_name, 
                aggregation_strategy="simple",
                device=device,
                # Reduce memory usage
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"NER model loaded successfully on {'GPU' if device >= 0 else 'CPU'}")
        
        # Process with timeout protection
        preds = NER_PIPE(text)
        names = []
        for p in preds:
            if p.get("entity_group", "").upper() in {"PER","PERSON"} and float(p.get("score", 0)) >= thr:
                w = (p.get("word") or "").strip()
                if w and w not in names:
                    names.append(w)
        return names
        
    except Exception as e:
        print(f"Warning: NER processing failed: {e}")
        print("Falling back to rule-based extraction only")
        return []

def exp_decay(delta_ms: int, tau_ms: int) -> float:
    if delta_ms <= 0:
        return 1.0
    return math.exp(-float(delta_ms)/max(1, tau_ms))

def fuzzy_match(name: str, roster: List[str], thr: int = 85) -> Optional[str]:
    if not roster or process is None:
        return None
    cand = process.extractOne(name, roster, scorer=fuzz.WRatio)
    if cand and cand[1] >= thr:
        return cand[0]
    return None

def uniq(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# Weights for decisions
def default_weights():
    return {
        # kanıt
        "w_self": 3.0, "w_addr": 0.6, "w_attr": 0.5, "w_ner": 0.4, "w_roster": 0.5,
        "w_addr_strong": -2.5,
        # candidate_speakers tipli kanıt
        "w_cand_self": 0.8, "w_cand_addr": -0.6, "w_cand_dcont": 0.6, "w_cand_map": 0.4,
        # pencere/süreklilik
        "w_prev": 0.5, "w_next": 0.3, "w_cont": 0.8, "w_spk_cont": 0.5, "w_cur": 1.0,
        "cont_min_score": 1.0
    }

# Evidence score and candidate collection
def gather_candidates(item: Dict, events: Dict[str, List[str]], ner_names: List[str]) -> List[str]:
    cands = []
    for k in ("self_id","addresssee","attribution","addresssee_strong"):
        cands.extend(events.get(k) or [])
    
    # Handle mentioned_attendees - can be list of dicts or list of strings
    for m in item.get("mentioned_attendees") or []:
        if isinstance(m, dict):
            name = (m.get("name") or "").strip()
        elif isinstance(m, str):
            name = m.strip()
        else:
            continue
        if name: cands.append(name)
    
    # Handle candidate_speakers - can be list of dicts or list of strings
    for cs in item.get("candidate_speakers") or []:
        if isinstance(cs, dict):
            name = (cs.get("name") or "").strip()
        elif isinstance(cs, str):
            name = cs.strip()
        else:
            continue
        if name: cands.append(name)
    
    cands.extend(ner_names or [])
    return uniq([c for c in cands if c])

def evidence_scores(item: Dict, cand: str, events: Dict[str, List[str]], ner_names: List[str],
                    roster: List[str], W: Dict[str,float]) -> float:
    sc = 0.0
    if cand in (events.get("self_id") or []):         sc += W["w_self"]
    if cand in (events.get("addresssee") or []):      sc += W["w_addr"]
    if cand in (events.get("attribution") or []):     sc += W["w_attr"]
    if cand in (events.get("addresssee_strong") or []): sc += W["w_addr_strong"]  # negatif
    if cand in ner_names:                              sc += W["w_ner"]
    for cs in item.get("candidate_speakers") or []:
        # Handle both dict and string formats
        if isinstance(cs, dict):
            nm = (cs.get("name") or "").strip()
            mtypes = cs.get("match_type") or []
        elif isinstance(cs, str):
            nm = cs.strip()
            mtypes = []  # No match types for string format
        else:
            continue
            
        if nm != cand:
            continue
        if "self_identification" in mtypes:    sc += W["w_cand_self"]
        if "addressed_person" in mtypes:       sc += W["w_cand_addr"]
        if "diarization_continuity" in mtypes: sc += W["w_cand_dcont"]
        if "name_in_speaker_map" in mtypes:    sc += W["w_cand_map"]
    if fuzzy_match(cand, roster):
        sc += W["w_roster"]
    return sc

# Segment Enrichment
def enrich_items(items: List[Dict], roster: List[str], ner_model: Optional[str], W: Dict[str,float]):
    enriched = []
    for it in items:
        text = it.get("text_corrected") or ""
        events = extract_events(text)
        ner_names = ner_persons(text, ner_model)
        start_ms = int(round(float(it.get("start", 0.0)) * 1000))
        end_ms   = int(round(float(it.get("end", 0.0)) * 1000))
        cands = gather_candidates(it, events, ner_names)
        enriched.append({
            "orig": it, "events": events, "ner_names": ner_names, "cands": cands,
            "start_ms": start_ms, "end_ms": end_ms
        })
    return enriched

# Canonical name selection
def build_speaker_canonical_map(enriched, roster: List[str], W: Dict[str,float],
                                strong_thr: float = 1.5, margin: float = 0.6, min_dur_ms: int = 3000):
    """
    Her speaker için güçlü kanıt toplamını hesapla, kanonik adı seç.
    - Yalnızca güçlü sinyaller: self_id, cand_self, diarization_continuity(+prev aynı ad),
      roster boost (küçük), addresssee_strong NEGATİF.
    - Toplam konuşma süresi çok kısa olan speaker'lar filtrelenebilir.
    """
    per_spk_stats = defaultdict(lambda: {"dur":0, "scores":Counter()})
    for e in enriched:
        spk = (e["orig"].get("speaker") or "").strip() or "spk_?"
        dur = max(0, e["end_ms"] - e["start_ms"])
        per_spk_stats[spk]["dur"] += dur
        # strong signals
        ev = e["events"]
        strong_names = set(ev.get("self_id") or [])
        # candidate_speakers type self_identification / diarization_continuity
        for cs in e["orig"].get("candidate_speakers") or []:
            name = (cs.get("name") or "").strip()
            if not name:
                continue
            mtypes = cs.get("match_type") or []
            if "self_identification" in mtypes or "diarization_continuity" in mtypes:
                strong_names.add(name)
        # Negative: strong
        for name in (ev.get("addresssee_strong") or []):
            per_spk_stats[spk]["scores"][name] += W["w_addr_strong"]  # negatif katkı
        # Positive Contribution
        for name in strong_names:
            gain = W["w_self"]  # self_id weight
            if fuzzy_match(name, roster):
                gain += 0.2  # attendee contribution
            per_spk_stats[spk]["scores"][name] += gain

    speaker_id_map = {}
    for spk, st in per_spk_stats.items():
        if st["dur"] < min_dur_ms:
            continue
        if not st["scores"]:
            continue
        # best and second score difference
        best = st["scores"].most_common(2)
        name1, score1 = best[0][0], float(best[0][1])
        score2 = float(best[1][1]) if len(best) > 1 else -1e9
        if score1 >= strong_thr and (score1 - score2) >= margin:
            speaker_id_map[spk] = {"name": name1, "score": score1}
    participants = [{"speaker_id": s, "name": v["name"], "score": v["score"], "duration_ms": per_spk_stats[s]["dur"]}
                    for s,v in speaker_id_map.items()]
    return speaker_id_map, participants, per_spk_stats

# pass 2
def assign_with_speaker_map(enriched, speaker_id_map, roster: List[str], W: Dict[str,float],
                            tau_ms: int, threshold: float):
    prev_name=None; prev_score=0.0
    out=[]
    for i in range(len(enriched)):
        cur = enriched[i]
        i_prev = i-1 if i-1>=0 else None
        i_next = i+1 if i+1<len(enriched) else None

        # default speaker map
        spk = (cur["orig"].get("speaker") or "").strip()
        map_name = speaker_id_map.get(spk, {}).get("name")

        # Check the strong counter-evidence in the segment
        ev = cur["events"]
        strong_conflict = False
        # For example, if self_id is a different name or if addressee_strong exists and contradicts with map_name
        sid = set(ev.get("self_id") or [])
        addrS = set(ev.get("addresssee_strong") or [])
        if sid and (map_name is None or (map_name not in sid)):
            strong_conflict = True
        if map_name and (map_name in addrS):
            strong_conflict = True

        # Time decays
        t_curr = cur["start_ms"]
        dec_prev = exp_decay(t_curr - (enriched[i_prev]["end_ms"] if i_prev is not None else t_curr), tau_ms)
        dec_next = exp_decay((enriched[i_next]["start_ms"] if i_next is not None else t_curr) - cur["end_ms"], tau_ms)

        # Candidate set and window score (when needed)
        cand_set = set(cur["cands"])
        if i_prev is not None: cand_set.update(enriched[i_prev]["cands"])
        if i_next is not None: cand_set.update(enriched[i_next]["cands"])
        def ev_score(j, c):
            e = enriched[j]
            return evidence_scores(e["orig"], c, e["events"], e["ner_names"], roster, W)

        scores={}
        for c in cand_set:
            s_cur  = ev_score(i, c) * W["w_cur"]
            s_prev = (ev_score(i_prev, c) if i_prev is not None else 0.0) * W["w_prev"] * dec_prev
            s_next = (ev_score(i_next, c) if i_next is not None else 0.0) * W["w_next"] * dec_next
            s_cont = (W["w_cont"] if (prev_name==c and prev_score>=W["cont_min_score"]) else 0.0)
            s_spkc = 0.0
            if i_prev is not None:
                prev_spk_id = (enriched[i_prev]["orig"].get("speaker") or "").strip()
                if prev_spk_id and spk and prev_spk_id==spk and prev_name==c and prev_score>=W["cont_min_score"]:
                    s_spkc = W["w_spk_cont"]
            scores[c] = s_cur + s_prev + s_next + s_cont + s_spkc

        # Decision:
        # - If no conflict and map_name exists -> assign it directly.
        # - If conflict exists -> select with window score; if below threshold, None.
        assigned_name=None; assigned_score=0.0; reason=""
        if map_name and not strong_conflict:
            assigned_name = map_name
            assigned_score = max(W["w_self"], 1.2)  # kanonik atamanın taban gücü
            reason = f"speaker_map('{spk}')"
        else:
            if scores:
                assigned_name, assigned_score = max(scores.items(), key=lambda kv: kv[1])
                if assigned_score < threshold:
                    assigned_name=None
            reason = f"windowed{'_conflict' if strong_conflict else ''}"

        conf = None
        if assigned_name is not None:
            conf = 1.0 - 1.0/(1.0 + max(0.0, assigned_score))

        # Base strength of canonical assignment
        it = cur["orig"]
        it["start_ms"]=cur["start_ms"]; it["end_ms"]=cur["end_ms"]
        it["events"]=cur["events"]; it["ner_names"]=cur["ner_names"]
        it["names_union"]=sorted(set(cur["cands"]))
        it["assigned_name"]=assigned_name
        it["assigned_confidence"]=conf
        it["assigned_reason"]=reason
        # debug
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        it["name_scores"]=[{"name": n, "score": s} for n,s in top3]

        if assigned_name is not None:
            prev_name=assigned_name; prev_score=assigned_score
        else:
            prev_score=0.0
        out.append(it)
    return out

def main(input_json_path: str,
         output_json_path: str,
         attendees = [],
         ner_model="Davlan/xlm-roberta-base-ner-hrl",
         tau_ms=90000,
         threshold=0.4,
         spk_strong_thr=1.5,
         spk_margin=0.6,
         spk_min_dur_ms=3000):

    W = default_weights()
    with open(input_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
        if not isinstance(items, list):
            raise RuntimeError("Input must be a JSON array")

    ner_model = ner_model if ner_model.strip() else None

    # Pass 1: enrich
    enriched = enrich_items(items, attendees, ner_model, W)

    # Speaker canonical map
    speaker_id_map, participants, per_spk_stats = build_speaker_canonical_map(
        enriched, attendees, W,
        strong_thr=spk_strong_thr, margin=spk_margin, min_dur_ms=spk_min_dur_ms
    )

    # Pass 2: assign with speaker map fallback to windowed
    out_items = assign_with_speaker_map(enriched, speaker_id_map, attendees, W, tau_ms, threshold)

    # Özet meta: katılımcı sayısı ve map
    meta = {
        "participants_count": len({(s if s else 'spk_?') for s in [ (e['orig'].get('speaker') or '').strip() for e in enriched ]}),
        "speakers_detected": sorted(list({(s if s else 'spk_?') for s in [ (e['orig'].get('speaker') or '').strip() for e in enriched ]})),
        "speaker_id_map": speaker_id_map,
        "participants_summary": participants
    }
    # Meta’yı ilk objeye ekleyelim (isterseniz ayrı bir dosyaya da yazabilirsiniz)
    if out_items:
        out_items[0]["_meta_assignment_summary"] = meta

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_items, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {output_json_path}  items={len(out_items)} | unique_speakers={meta['participants_count']}")

def apply_name_extraction_to_segments(segments: List[Dict], attendees: List[str] = None, 
                                      ner_model: Optional[str] = None,
                                      tau_ms: int = 90000, threshold: float = 0.4,
                                      spk_strong_thr: float = 1.5, spk_margin: float = 0.6,
                                      spk_min_dur_ms: int = 3000) -> List[Dict]:
    """
    Apply rule-based name extraction to transcript segments and add extracted_names field.
    
    Args:
        segments: List of transcript segments from the pipeline
        attendees: List of attendee names
        ner_model: Optional NER model name
        Other parameters: Algorithm tuning parameters
    
    Returns:
        List of segments with added extracted_names field
    """
    if not segments:
        return segments
    
    # Default attendees if none provided
    if not attendees:
        attendees = []
    
    # Create a copy of segments to avoid modifying the original
    items = []
    for seg in segments:
        # Map current pipeline format to rule-based algorithm format
        item = {
            "text_corrected": seg.get("text_corrected", seg.get("text", "")),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker": seg.get("speaker", ""),
            "mentioned_attendees": seg.get("mentioned_attendees", []),
            "candidate_speakers": seg.get("candidate_speakers", [])
        }
        items.append(item)
    
    try:
        W = default_weights()
        
        # Pass 1: enrich
        enriched = enrich_items(items, attendees, ner_model, W)
        
        # Speaker canonical map
        speaker_id_map, participants, per_spk_stats = build_speaker_canonical_map(
            enriched, attendees, W,
            strong_thr=spk_strong_thr, margin=spk_margin, min_dur_ms=spk_min_dur_ms
        )
        
        # Pass 2: assign with speaker map fallback to windowed
        processed_items = assign_with_speaker_map(enriched, speaker_id_map, attendees, W, tau_ms, threshold)
        
        # Now add extracted_names field to original segments
        result_segments = []
        for i, seg in enumerate(segments):
            # Copy original segment
            new_seg = seg.copy()
            
                # Extract names from the processed item
            if i < len(processed_items):
                processed = processed_items[i]
                extracted_names = []
                
                # Priority 1: Add assigned name if present (highest confidence)
                assigned_name = processed.get("assigned_name")
                if assigned_name and assigned_name.strip():
                    extracted_names.append(assigned_name.strip())
                
                # Priority 2: Add unique names from events (rule-based extraction)
                events = processed.get("events", {})
                
                # Self-identification names (high confidence)
                for name in events.get("self_id", []):
                    if name and name.strip() and name.strip() not in extracted_names:
                        extracted_names.append(name.strip())
                
                # Attribution names (medium confidence)
                for name in events.get("attribution", []):
                    if name and name.strip() and name.strip() not in extracted_names:
                        extracted_names.append(name.strip())
                
                # Addressee names (lower confidence, but useful)
                for name in events.get("addresssee", []):
                    if name and name.strip() and name.strip() not in extracted_names:
                        # Filter out common greeting phrases that might be wrongly captured
                        clean_name = name.strip()
                        if not any(word in clean_name.lower() for word in ["teşekkürler", "merhaba", "selam", "thanks", "hello"]):
                            extracted_names.append(clean_name)
                
                # Strong addressee (mostly for filtering, but can include valid names)
                for name in events.get("addresssee_strong", []):
                    if name and name.strip() and name.strip() not in extracted_names:
                        clean_name = name.strip()
                        # Only add if it looks like a proper name (contains only letters and spaces)
                        if re.match(r'^[A-ZÇĞİÖŞÜa-zçğıöşü\s]+$', clean_name) and len(clean_name.split()) <= 3:
                            extracted_names.append(clean_name)
                
                # Priority 3: Add NER names (if available)
                ner_names = processed.get("ner_names", [])
                for name in ner_names:
                    if name and name.strip() and name.strip() not in extracted_names:
                        extracted_names.append(name.strip())
                
                # Filter out duplicates and clean names
                final_names = []
                for name in extracted_names:
                    # Basic name validation
                    if (name and 
                        len(name) > 1 and  # At least 2 characters
                        len(name) < 50 and  # Not too long
                        not name.isdigit() and  # Not just numbers
                        re.search(r'[A-ZÇĞİÖŞÜa-zçğıöşü]', name)):  # Contains at least one letter
                        final_names.append(name)
                
                new_seg["extracted_names"] = final_names
            else:
                new_seg["extracted_names"] = []
            
            result_segments.append(new_seg)
        
        return result_segments
        
    except Exception as e:
        print(f"Warning: Name extraction failed: {e}")
        # Return original segments with empty extracted_names
        result_segments = []
        for seg in segments:
            new_seg = seg.copy()
            new_seg["extracted_names"] = []
            result_segments.append(new_seg)
        return result_segments


if __name__ == "__main__":
    main(input_json_path="/content/transcript_V7.json",
         output_json_path="/content/named_transcript.json")