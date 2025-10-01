
import os
from typing import List, Dict, Any, Optional

from app.core.model_registry import models

try:
    from app.pipeline.config import CHUNK_LENGTH, CHUNK_OVERLAP
except Exception:
    CHUNK_LENGTH = 240.0
    CHUNK_OVERLAP = 4.0

try:
    from app.pipeline.ffmpeg_utils import cut_segment
except Exception as e:
    raise ImportError("ffmpeg_utils.cut_segment bulunamadı. İmzası: cut_segment(input_wav, output_wav, start_s, end_s, idx)") from e

# Import Replicate transcription
try:
    from app.pipeline.replicate_whisper_transcription import run_replicate_whisper_transcription_for_segments
    REPLICATE_TRANSCRIPTION_AVAILABLE = True
    print("✅ Replicate transcription available")
except ImportError as e:
    REPLICATE_TRANSCRIPTION_AVAILABLE = False
    print(f"⚠️  Replicate transcription not available: {e}")


def _hop() -> float:
    """Overlapped chunk'larda ofset = i * (CHUNK_LENGTH - CHUNK_OVERLAP)."""
    return float(CHUNK_LENGTH) - float(CHUNK_OVERLAP)


def _find_chunk_file(chunk_dir: str, chunk_index: int) -> Optional[str]:
    """Hem sıfır dolgulu hem düz isimlendirmeyi destekle."""
    cand1 = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}.wav")
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(chunk_dir, f"chunk_{chunk_index}.wav")
    if os.path.exists(cand2):
        return cand2
    return None


def run_whisper_transcription(segments: List[Dict[str, Any]], chunk_dir: str, language: str = "tr") -> List[Dict[str, Any]]:
    """
    Transcription using Replicate's Incredibly Fast Whisper as primary method,
    with fallback to original segment-based Whisper transcription.

    Parametreler
    -----------
    segments : List[Dict]
        Diarization çıktısı. En az 'start', 'end', 'speaker', 'chunk' alanları beklenir.
    chunk_dir : str
        chunk_XXXX.wav dosyalarının bulunduğu dizin.
    language : str
        Whisper dil kodu (varsayılan: "tr").

    Dönüş
    -----
    List[Dict] : Her öğe {start, end, speaker, text}
    """

    # Check if Replicate transcription is available and API token is configured
    use_replicate = REPLICATE_TRANSCRIPTION_AVAILABLE

    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if replicate_token == "r8_xxx..." or not replicate_token:
        print("⚠️  REPLICATE_API_TOKEN not configured properly, falling back to local Whisper")
        use_replicate = False

    # Try Replicate transcription first
    if use_replicate:
        try:
            print("🚀 Using Replicate Incredibly Fast Whisper for transcription...")
            return run_replicate_whisper_transcription_for_segments(
                segments=segments,
                chunk_dir=chunk_dir,
                language=language
            )
        except Exception as e:
            print(f"❌ Replicate transcription failed: {e}")
            print("🔄 Falling back to original Whisper transcription...")

    # Fallback to original segment-based Whisper transcription
    print("🔄 Using original segment-based Whisper transcription...")
    return run_original_whisper_transcription(segments, chunk_dir, language)


def run_original_whisper_transcription(segments: List[Dict[str, Any]], chunk_dir: str, language: str = "tr") -> List[Dict[str, Any]]:
    """
    Original segment-based Whisper transcription (renamed for clarity).
    This is the fallback method when Replicate is not available or fails.

    Diarization segmentleri için, ilgili chunk dosyasından FFmpeg ile kesit alıp Whisper ile transkribe eder.
    - Ofset hesabı overlap-aware'dır: offset = chunk_index * (CHUNK_LENGTH - CHUNK_OVERLAP).
    - start/end değerleri chunk yereline çevrilip aralık dışına taşma varsa kırpılır.
    - Her segment için tek bir geçici .wav üretilir.

    Parametreler
    -----------
    segments : List[Dict]
        Diarization çıktısı. En az 'start', 'end', 'speaker', 'chunk' alanları beklenir.
    chunk_dir : str
        chunk_XXXX.wav dosyalarının bulunduğu dizin.
    language : str
        Whisper dil kodu (varsayılan: "tr").

    Dönüş
    -----
    List[Dict] : Her öğe {start, end, speaker, chunk, text}
    """
    results: List[Dict[str, Any]] = []
    tmp_paths: List[str] = []

    hop = _hop()
    if hop <= 0:
        raise ValueError("CHUNK_LENGTH ve CHUNK_OVERLAP değerleri hatalı: hop <= 0.")

    for i, seg in enumerate(segments):
        # Zorunlu alanlar
        if not all(k in seg for k in ("start", "end", "chunk")):
            # Eksik segmenti atla
            continue

        try:
            chunk_index = int(seg["chunk"])
            g_start = float(seg["start"])  # global
            g_end = float(seg["end"])      # global
        except Exception:
            continue

        # İlgili chunk dosyası
        chunk_file = _find_chunk_file(chunk_dir, chunk_index)
        if not chunk_file:
            # Bulunamazsa atla
            continue

        # Global -> yerel zaman
        offset = chunk_index * hop
        local_start = max(0.0, g_start - offset)
        local_end = max(0.0, g_end - offset)

        # Güvenli kırpma: en az 0.10s, en fazla CHUNK_LENGTH
        # (Son chunk daha kısa olabilir; ffmpeg kısa aralıkları zaten tolere eder.)
        if local_end <= local_start:
            # Çok kısa veya ters aralık -> minimum 0.10s dene
            local_end = local_start + 0.10

        # Kesit dosya yolu
        segment_file = os.path.join(chunk_dir, f"segment_{i:06d}.wav")
        tmp_paths.append(segment_file)

        # FFmpeg kesimi
        try:
            cut_segment(chunk_file, segment_file, local_start, local_end, i)
        except Exception as e:
            # Kesim başarısızsa bu segmenti atla
            continue

        # Whisper
        try:
            transcript = models.whisper.transcribe(segment_file, language=language)
            if isinstance(transcript, dict):
                text = (transcript.get("text") or "").strip()
            else:
                text = str(transcript).strip()

            results.append({
                "start": g_start,
                "end": g_end,
                "speaker": seg.get("speaker"),
                "chunk": chunk_index,
                "text": text
            })
        except Exception:
            # Bu segmenti atla
            continue

    # Geçicileri sil
    if os.getenv("DELETE_SEGMENTS_AFTER_TRANSCRIBE", "false").lower() == "true":
        for p in tmp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    return results
