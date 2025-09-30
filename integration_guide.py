#!/usr/bin/env python3
"""
🎯 MEVCUT KODUNUZA DECORATOR ENTEGRASYONU
========================================

Bu dosya, mevcut transcript pipeline'ınıza monitoring nasıl entegre edeceğinizi gösterir.
"""

# ============================================================================
# 1. MEVCUT app/main.py DOSYANIZI GÜNCELLEYİN
# ============================================================================

"""
ÖNCE (Mevcut main.py):

from fastapi import FastAPI, HTTPException
from app.schemas import TranscribeRequest, TranscribeResponse
from app.pipeline.pipeline import run_transcript_pipeline

app = FastAPI(title="Arya HI Transcript Extraction API")

@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_video(request: TranscribeRequest):
    try:
        result = run_transcript_pipeline(
            mp4_path=request.video_path,
            user_id=request.user_id,
            attendees=request.attendees,
            meeting_id=request.meeting_id
        )
        return {"json_output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

# ============================================================================
# SONRA (Monitoring ile güncellenmiş main.py):
# ============================================================================

from fastapi import FastAPI, HTTPException
from app.schemas import TranscribeRequest, TranscribeResponse
from app.pipeline.pipeline import run_transcript_pipeline

# 🚀 SADECE BU IMPORT'U EKLEYİN!
from transcript_monitoring import transcript_monitor, get_monitoring_stats, health_check

app = FastAPI(title="Arya HI Transcript Extraction API")

@app.get("/health")
def health_check_endpoint():
    return {"status": "healthy", "service": "transcript-api"}

# 🚀 YENİ: Monitoring health check endpoint
@app.get("/monitoring/health")
async def monitoring_health():
    """Monitoring sistem durumunu kontrol et."""
    health_status = await health_check()
    return health_status

# 🚀 YENİ: Monitoring statistics endpoint
@app.get("/monitoring/stats")
def monitoring_stats():
    """Monitoring istatistiklerini getir."""
    return get_monitoring_stats()

# 🚀 SADECE DECORATOR EKLEYİN - BAŞKA HİÇBİR ŞEY DEĞİŞMEDİ!
@app.post("/transcribe", response_model=TranscribeResponse)
@transcript_monitor(
    user_context=lambda request: request.user_id,
    session_context=lambda request: request.meeting_id,
    track_audio_metadata=True,
    send_to_vertex=True  # .env'de SEND_TO_VERTEX=true olduğunda aktif
)
async def transcribe_video(request: TranscribeRequest):
    """
    Video transkript işlemi - şimdi otomatik monitoring ile!

    Decorator otomatik olarak:
    - Performance metrics toplar (latency, processing time)
    - User ID ve meeting ID'yi track eder
    - Background'da Vertex AI'ya gönderir
    - Error handling yapar
    """
    try:
        # 🚀 MEVCUT KODUNUZ HİÇ DEĞİŞMEDİ!
        result = await run_transcript_pipeline_async(  # Async yaptık
            mp4_path=request.video_path,
            user_id=request.user_id,
            attendees=request.attendees,
            meeting_id=request.meeting_id
        )
        return {"json_output": result}
    except Exception as e:
        # Decorator otomatik olarak error'u track eder
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 2. app/pipeline/pipeline.py DOSYANIZI GÜNCELLEYİN
# ============================================================================

"""
MEVCUT pipeline.py dosyanızda sadece şunları yapın:

1. Import ekleyin:
"""

# Dosyanın başına ekleyin:
from transcript_monitoring import transcript_monitor

"""
2. Ana fonksiyonu async yapın ve decorator ekleyin:
"""

# ÖNCE:
# def run_transcript_pipeline(mp4_path: str, output_root: str = DEFAULT_OUTPUT_ROOT,
#                            user_id: str = "anonymous", attendees: Optional[List[str]] = None,
#                            meeting_id: Optional[str] = None) -> dict:

# SONRA:
@transcript_monitor(
    user_context=lambda *args, **kwargs: kwargs.get('user_id', 'anonymous'),
    session_context=lambda *args, **kwargs: kwargs.get('meeting_id'),
    track_audio_metadata=True,
    custom_config={
        "track_performance_metrics": True,
        "track_quality_metrics": True,
        "offline_mode": True  # İlk test için offline
    }
)
async def run_transcript_pipeline(mp4_path: str, output_root: str = None,
                                 user_id: str = "anonymous",
                                 attendees: Optional[List[str]] = None,
                                 meeting_id: Optional[str] = None) -> dict:
    """
    Şimdi bu fonksiyon otomatik olarak:
    - Her çağrı için unique request ID oluşturur
    - Audio file metadata'sını toplar
    - Her aşamanın süresini ölçer
    - User ve session bazında analytics toplar
    - Background'da Vertex AI'ya gönderir
    """

    # 🚀 MEVCUT KODUNUZ TAMAMEN AYNI KALIYOR!
    # Sadece async/await ekleyeceksiniz gerektiğinde

    session_id = uuid4().hex
    # ... mevcut kodunuz ...

    return final_result


# ============================================================================
# 3. ASYNC DESTEĞİ İÇİN KÜÇÜK DEĞİŞİKLİKLER
# ============================================================================

"""
Bazı sync fonksiyonları async yapmak gerekebilir. İşte örnekler:
"""

# Gemini API çağrıları zaten async ise sorun yok
# Diğer sync fonksiyonlar için:

async def run_transcript_pipeline_async(*args, **kwargs):
    """Wrapper to make sync pipeline async."""
    import asyncio

    # CPU-intensive işlemler için thread pool kullan
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_transcript_pipeline_sync, *args, **kwargs)

def run_transcript_pipeline_sync(*args, **kwargs):
    """Mevcut sync pipeline'ınız."""
    # Mevcut kodunuz burada kalır
    pass


# ============================================================================
# 4. ADVANCED: GRANÜLEr MONİTORİNG (İSTEĞE BAĞLI)
# ============================================================================

"""
Daha detaylı monitoring istiyorsanız, pipeline içinde context manager kullanın:
"""

from transcript_monitoring import TranscriptMonitor

async def advanced_transcript_pipeline(mp4_path: str, user_id: str, **kwargs):
    """Pipeline içinde detaylı monitoring."""

    async with TranscriptMonitor(user_id=user_id, session_id=kwargs.get('meeting_id')) as monitor:

        # 1. Audio processing phase
        monitor.mark_audio_processing_start()
        monitor.add_audio_metadata(file_path=mp4_path)

        wav_path = await convert_mp4_to_wav(mp4_path)
        chunks = await split_wav_into_chunks_v2(wav_path)

        monitor.mark_audio_processing_end()

        # 2. Diarization phase
        monitor.add_custom_metadata(phase="diarization", chunk_count=len(chunks))

        diarized_chunks = await diarize_chunks_with_global_ids_union(chunks)

        # 3. Transcription phase
        monitor.mark_model_inference_start()
        monitor.add_custom_metadata(phase="transcription", model="whisper")

        transcribed_segments = await run_whisper_transcription(diarized_chunks)

        # 4. Correction phase (Gemini)
        monitor.add_custom_metadata(phase="correction", model="gemini")

        corrected_segments = await gemini_correct_and_infer_segments_async_v3(transcribed_segments)

        monitor.mark_model_inference_end()

        # Final result
        final_result = {
            "segments": corrected_segments,
            "total_duration": sum(s['duration'] for s in corrected_segments),
            "word_count": sum(len(s['text'].split()) for s in corrected_segments)
        }

        monitor.set_result(final_result)

        return final_result


# ============================================================================
# 5. ÖRNEK KULLANIM - TEST EDİN
# ============================================================================

async def test_monitoring_integration():
    """Test your monitoring integration."""

    # Simulate a request
    from app.schemas import TranscribeRequest

    test_request = TranscribeRequest(
        video_path="/path/to/test/video.mp4",
        user_id="test_user_123",
        meeting_id="meeting_456",
        attendees=["Alice", "Bob"]
    )

    print("🎬 Testing monitored transcription...")

    # Bu çağrı otomatik olarak monitor edilecek
    result = await transcribe_video(test_request)

    print(f"✅ Result: {result}")

    # Monitoring stats'a bak
    import asyncio
    await asyncio.sleep(2)  # Background processing için bekle

    stats = get_monitoring_stats()
    print(f"📊 Monitoring stats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_monitoring_integration())


# ============================================================================
# 6. ÖZET - YAPMANIZ GEREKENLER
# ============================================================================

"""
✅ CHECKLIST:

1. app/main.py'a import ekle:
   from transcript_monitoring import transcript_monitor, get_monitoring_stats, health_check

2. transcribe_video fonksiyonuna decorator ekle:
   @transcript_monitor(user_context=lambda request: request.user_id, ...)

3. app/pipeline/pipeline.py'da run_transcript_pipeline'a decorator ekle:
   @transcript_monitor(user_context=lambda **kw: kw.get('user_id'), ...)

4. Fonksiyonları async yap (gerekirse):
   async def run_transcript_pipeline(...)

5. .env'de ayarları kontrol et:
   MONITORING_ENABLED=true
   SEND_TO_VERTEX=false (ilk test için)
   MONITORING_OFFLINE_MODE=true

6. Test et:
   python integration_guide.py

TAMAM! Artık mevcut pipeline'ınız otomatik monitoring'e sahip! 🎉
"""