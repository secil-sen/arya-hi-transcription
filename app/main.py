from fastapi import FastAPI, HTTPException
from app.schemas import TranscribeRequest, TranscribeResponse
from app.core.warnings_config import setup_production_environment
from app.core.error_handler import (
    configure_logging, ErrorHandler, CommonErrors, TranscriptException
)
from app.services.transcript_service import TranscriptServiceFactory, TranscriptRequest

# Configure production environment early
setup_production_environment()

# Configure logging
configure_logging(log_level="INFO")

# 🚀 MONITORING IMPORT - Otomatik metrics tracking için
try:
    from transcript_monitoring import transcript_monitor, get_monitoring_stats, health_check, start_monitoring, stop_monitoring
    MONITORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Monitoring not available: {e}")
    MONITORING_AVAILABLE = False

    # Dummy decorator when monitoring not available
    def transcript_monitor(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    async def health_check():
        return {"healthy": False, "error": "Monitoring not available"}

    def get_monitoring_stats():
        return {"error": "Monitoring not available"}

    async def start_monitoring():
        pass

    async def stop_monitoring():
        pass

# Import pipeline with fallback
try:
    from app.pipeline.pipeline import run_transcript_pipeline
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Pipeline not available: {e}")
    PIPELINE_AVAILABLE = False

    def run_transcript_pipeline(*args, **kwargs):
        raise HTTPException(status_code=503, detail="Pipeline dependencies not available")

app = FastAPI(title="Arya HI Transcript Extraction API")

# 🚀 STARTUP EVENT - Monitoring sistemi otomatik başlatma
@app.on_event("startup")
async def startup_event():
    """FastAPI başladığında monitoring sistemi başlat."""
    if MONITORING_AVAILABLE:
        print("🚀 Starting monitoring system...")
        await start_monitoring()
        print("✅ Monitoring system started!")
    else:
        print("⚠️ Monitoring system not available - running without monitoring")

    if not PIPELINE_AVAILABLE:
        print("⚠️ Pipeline dependencies not available - /transcribe endpoint will return 503")

# 🚀 SHUTDOWN EVENT - Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI kapanırken monitoring sistemi güvenli kapatma."""
    if MONITORING_AVAILABLE:
        print("🛑 Stopping monitoring system...")
        await stop_monitoring()
        print("✅ Monitoring system stopped!")

@app.get("/health")
def health_check_endpoint():
    return {"status": "healthy", "service": "transcript-api"}

# 🚀 YENİ: Monitoring health check endpoint
@app.get("/monitoring/health")
async def monitoring_health():
    """Monitoring sistem durumunu kontrol et."""
    try:
        health_status = await health_check()
        return health_status
    except Exception as e:
        return {
            "healthy": False,
            "error": f"Health check failed: {str(e)}",
            "details": "Monitoring system may not be initialized. Check startup logs."
        }

# 🚀 YENİ: Monitoring statistics endpoint
@app.get("/monitoring/stats")
def monitoring_stats():
    """Monitoring istatistiklerini getir."""
    try:
        stats = get_monitoring_stats()
        return stats
    except Exception as e:
        return {
            "error": f"Stats collection failed: {str(e)}",
            "details": "Monitoring system may not be initialized. Check startup logs."
        }

# 🚀 DECORATOR EKLENDİ - Otomatik monitoring aktif!
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
        result = await run_transcript_pipeline_async(
            mp4_path=request.video_path,
            user_id=request.user_id,
            attendees=request.attendees,
            meeting_id=request.meeting_id
        )
        return {"json_output": result}
    except Exception as e:
        # Decorator otomatik olarak error'u track eder
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 WRAPPER FUNCTION - Pipeline artık async, direkt çağır
async def run_transcript_pipeline_async(mp4_path: str, user_id: str = "anonymous",
                                       attendees=None, meeting_id=None):
    """Pipeline artık async, direkt çağır."""
    if PIPELINE_AVAILABLE:
        return await run_transcript_pipeline(
            mp4_path=mp4_path,
            user_id=user_id,
            attendees=attendees,
            meeting_id=meeting_id
        )
    else:
        raise HTTPException(status_code=503, detail="Pipeline dependencies not available")