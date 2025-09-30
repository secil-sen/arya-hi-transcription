# Transcript Monitoring System

Bu paket, Google Cloud Vertex AI ile entegre olarak transkript servislerinizi izlemek için geliştirilmiş bir monitoring sistemidir. Decorator pattern ve context manager ile minimal kod değişikliği ile mevcut pipeline'ınıza seamless entegrasyon sağlar.

## 🚀 Özellikler

- **Seamless Integration**: Decorator pattern ile mevcut kodunuzu minimal değiştirerek monitoring aktif
- **Performance Tracking**: Request latency, audio processing time, model inference time
- **Quality Metrics**: Transcript uzunluğu, confidence scores, token sayısı
- **Business Analytics**: User bazında usage tracking, cost monitoring
- **Background Processing**: Asenkron metrics gönderimi (ana request'i yavaşlatmaz)
- **Circuit Breaker**: Vertex AI down olsa bile servis çalışmaya devam eder
- **Offline Mode**: Metrics offline kaydedilip sonra gönderilebilir
- **Auto-retry**: Failed requests için exponential backoff ile retry

## 📦 Kurulum

```bash
# Monitoring dependencies ile birlikte
poetry install --extras monitoring

# Sadece core dependencies
poetry install
```

## ⚙️ Yapılandırma

`.env` dosyanızı güncelleyin:

```env
# Vertex AI Configuration
VERTEX_AI_PROJECT_ID=your-gcp-project-id
VERTEX_AI_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Monitoring Settings
MONITORING_ENABLED=true
SEND_TO_VERTEX=true
MONITORING_BATCH_SIZE=10
```

## 🎯 Kullanım Örnekleri

### 1. Decorator Pattern (Önerilen)

```python
from transcript_monitoring import transcript_monitor

@transcript_monitor(
    user_context=lambda *args, **kwargs: kwargs.get('user_id'),
    track_audio_metadata=True,
    send_to_vertex=True
)
async def transcribe_audio(audio_data: bytes, user_id: str) -> dict:
    # Mevcut transkript kodunuz - hiçbir şey değişmiyor!
    result = await gemini_api.transcribe(audio_data)
    return {
        "transcript": result.text,
        "confidence": result.confidence
    }

# Kullanım
result = await transcribe_audio(audio_file, user_id="user123")
```

### 2. Context Manager Pattern

```python
from transcript_monitoring import TranscriptMonitor

async def advanced_transcribe(audio_data: bytes, user_id: str):
    async with TranscriptMonitor(user_id=user_id) as monitor:
        # Fine-grained control
        monitor.mark_audio_processing_start()

        # Audio preprocessing
        processed = await preprocess_audio(audio_data)
        monitor.add_audio_metadata(
            duration_seconds=30.5,
            format="wav",
            file_size_bytes=len(audio_data)
        )

        monitor.mark_audio_processing_end()
        monitor.mark_model_inference_start()

        # Gemini API call
        result = await gemini_api.transcribe(processed)

        monitor.mark_model_inference_end()
        monitor.set_result(result)

        return result
```

### 3. Mevcut Pipeline'ınıza Entegrasyon

```python
# Sadece decorator ekleyerek mevcut kodunuzu monitor edin
@transcript_monitor(
    user_context=lambda audio, user, **kw: user,
    track_audio_metadata=True
)
async def your_existing_function(audio_file: bytes, user_id: str):
    # Mevcut pipeline kodunuz - değişmiyor
    step1 = await diarization_step(audio_file)
    step2 = await transcription_step(step1)
    step3 = await post_processing_step(step2)
    return step3
```

## 📊 Metrics ve Analytics

### Performance Metrics
- **Total Duration**: Request başından sonuna kadar geçen süre
- **Audio Processing Time**: Audio preprocessing süresi
- **Model Inference Time**: Gemini API çağrısı süresi
- **Pipeline Duration**: Toplam pipeline süresi

### Quality Metrics
- **Transcript Length**: Transcript uzunluğu (karakter)
- **Word Count**: Kelime sayısı
- **Confidence Scores**: Gemini'den gelen confidence değerleri
- **Average Confidence**: Ortalama confidence skoru

### Business Metrics
- **User Analytics**: User bazında kullanım istatistikleri
- **Session Tracking**: Session bazında performance
- **API Call Counts**: API çağrı sayıları
- **Cost Tracking**: Maliyet takibi

## 🔧 Advanced Configuration

### Custom Configuration

```python
from transcript_monitoring import transcript_monitor

@transcript_monitor(
    project_id="my-custom-project",
    user_context=lambda *args, **kwargs: extract_user_from_jwt(kwargs.get('token')),
    custom_config={
        "batch_size": 5,
        "send_to_vertex": True,
        "offline_mode": False
    }
)
async def my_function():
    pass
```

### Error Handling

```python
@transcript_monitor(user_context=lambda **kw: kw.get('user_id'))
async def safe_transcribe(audio_data: bytes, user_id: str):
    try:
        if not audio_data:
            raise ValueError("Audio required")

        return await process_audio(audio_data)
    except Exception as e:
        # Hata otomatik olarak metrics'e kaydedilir
        raise
```

## 📈 Dashboard ve Monitoring

Vertex AI Console'da aşağıdaki metrikleri görüntüleyebilirsiniz:

- **Real-time Performance**: Latency, throughput graphs
- **Quality Trends**: Confidence score trends over time
- **User Analytics**: User başına usage patterns
- **Error Monitoring**: Error rates ve types
- **Cost Analytics**: API usage ve cost optimization

## 🛠️ Maintenance

### Health Check

```python
from transcript_monitoring import health_check, get_monitoring_stats

# Health check
status = await health_check()
print(f"System healthy: {status['healthy']}")

# Statistics
stats = get_monitoring_stats()
print(f"Metrics queued: {stats['background_sender']['queue_size']}")
```

### Graceful Shutdown

```python
from transcript_monitoring import stop_monitoring

# Uygulama kapatılırken
await stop_monitoring()  # Pending metrics'leri gönderir
```

## 🔒 Security ve Privacy

- **No Data Storage**: Audio data asla saklanmaz, sadece metadata
- **Secure Transmission**: TLS ile güvenli Vertex AI iletişimi
- **Configurable Privacy**: Hangi metriklerin toplanacağını kontrol edebilirsiniz
- **Offline Mode**: İnternet bağlantısı olmasa bile çalışır

## 🚨 Troubleshooting

### Common Issues

1. **Vertex AI Connection Errors**
   ```bash
   # Check credentials
   gcloud auth application-default login
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
   ```

2. **Circuit Breaker OPEN**
   ```python
   # Check Vertex AI status
   status = await health_check()
   print(status['vertex_ai'])  # Should be True
   ```

3. **Missing Dependencies**
   ```bash
   poetry install --extras monitoring
   ```

## 📋 Best Practices

1. **Use Decorator Pattern** for new functions
2. **Use Context Manager** for fine-grained control
3. **Enable Offline Mode** during development
4. **Monitor Queue Sizes** in production
5. **Set Appropriate Batch Sizes** based on traffic
6. **Use Circuit Breaker** for resilience

## 🎁 Integration Examples

Projenizde `examples/monitoring_examples.py` dosyasına bakarak detaylı kullanım örneklerini inceleyebilirsiniz.

## 📞 Support

- **Issues**: GitHub issues açabilirsiniz
- **Documentation**: Bu README ve kod içi docstring'ler
- **Examples**: `examples/` klasöründeki örnekler