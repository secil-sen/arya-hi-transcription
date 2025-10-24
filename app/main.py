from fastapi import FastAPI, HTTPException
from app.schemas import TranscribeRequest, TranscribeResponse, AutoAnalysisNotification
from app.core.warnings_config import setup_production_environment
from app.core.error_handler import (
    configure_logging, ErrorHandler, CommonErrors, TranscriptException
)
from app.services.transcript_service import TranscriptServiceFactory, TranscriptRequest
import asyncio
from typing import Dict, Optional

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

# In-memory cache for RAG auto analysis results
# Structure: {meeting_id: {"data": AutoAnalysisData, "event": asyncio.Event}}
rag_analysis_cache: Dict[str, Dict] = {}
RAG_WAIT_TIMEOUT = 30  # seconds

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

@app.post("/rag/notify")
async def rag_notify(notification: AutoAnalysisNotification):
    """
    Receive auto_analysis_ready notification from RAG service.
    Stores the result in cache and signals waiting transcribe requests.
    """
    meeting_id = notification.meeting_id

    print(f"📥 Received RAG auto_analysis for meeting_id: {meeting_id}")
    print(f"   - Notification ID: {notification.notification_id}")
    print(f"   - Processing time: {notification.auto_analysis.processing_time_ms}ms")
    print(f"   - Chunk count: {notification.auto_analysis.chunk_count}")

    # Store in cache
    if meeting_id in rag_analysis_cache:
        rag_analysis_cache[meeting_id]["data"] = notification.auto_analysis
        # Signal the waiting event
        rag_analysis_cache[meeting_id]["event"].set()
        print(f"✅ RAG result stored and event signaled for meeting_id: {meeting_id}")
    else:
        # If no one is waiting yet, still store it (they might call /transcribe later)
        rag_analysis_cache[meeting_id] = {
            "data": notification.auto_analysis,
            "event": asyncio.Event()
        }
        rag_analysis_cache[meeting_id]["event"].set()
        print(f"⚠️  No waiting request found, but stored RAG result for meeting_id: {meeting_id}")

    return {
        "status": "success",
        "meeting_id": meeting_id,
        "message": "Auto analysis received and cached"
    }

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

    Yeni: RAG auto_analysis sonucunu bekler ve birleştirip gönderir
    """
    try:
        meeting_id = request.meeting_id

        # Initialize cache entry for this meeting if not exists
        if meeting_id not in rag_analysis_cache:
            rag_analysis_cache[meeting_id] = {
                "data": None,
                "event": asyncio.Event()
            }
            print(f"🔄 Created cache entry for meeting_id: {meeting_id}")

        # Run transcript pipeline
        print(f"🎬 Starting transcript pipeline for meeting_id: {meeting_id}")
        result = await run_transcript_pipeline_async(
            mp4_path=request.video_path,
            user_id=request.user_id,
            attendees=request.attendees,
            meeting_id=request.meeting_id
        )
        print(f"✅ Transcript pipeline completed for meeting_id: {meeting_id}")

        # Wait for RAG auto_analysis result (with timeout)
        print(f"⏳ Waiting for RAG auto_analysis (timeout: {RAG_WAIT_TIMEOUT}s)...")
        try:
            await asyncio.wait_for(
                rag_analysis_cache[meeting_id]["event"].wait(),
                timeout=RAG_WAIT_TIMEOUT
            )

            # RAG result received
            auto_analysis_data = rag_analysis_cache[meeting_id]["data"]
            if auto_analysis_data:
                print(f"✅ RAG auto_analysis received for meeting_id: {meeting_id}")
                result["auto_analysis"] = auto_analysis_data.dict()
            else:
                print(f"⚠️  RAG cache entry exists but data is None for meeting_id: {meeting_id}")
                result["auto_analysis"] = None

        except asyncio.TimeoutError:
            print(f"⏰ Timeout waiting for RAG auto_analysis for meeting_id: {meeting_id}")
            result["auto_analysis"] = None

        # Cleanup cache entry
        if meeting_id in rag_analysis_cache:
            del rag_analysis_cache[meeting_id]
            print(f"🧹 Cleaned up cache entry for meeting_id: {meeting_id}")

        return {"json_output": result}

    except Exception as e:
        # Cleanup cache on error
        if request.meeting_id in rag_analysis_cache:
            del rag_analysis_cache[request.meeting_id]
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