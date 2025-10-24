# Frontend Payload Example - RAG Auto Analysis Integration

## Overview
Bu dokümanda `/transcribe` endpoint'inin RAG auto analysis ile birlikte döndüreceği response yapısı gösterilmektedir.

## Flow Diagram

```
1. Frontend → POST /transcribe (meeting_id ile)
2. Transcript Service → Transkript işlemi başlar
3. Transcript Service → RAG servisine notification gönderir (finalize_ready)
4. RAG Service → Auto analysis yapar
5. RAG Service → POST /rag/notify (auto_analysis_ready)
6. Transcript Service → Her ikisini birleştirir
7. Transcript Service → Frontend'e combined response döner
```

## Complete Response Example

### Scenario 1: RAG Analysis Received (Success Case)

```json
{
  "json_output": {
    "status": "success",
    "session_id": "abc123def456",
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "speaker": "spk_0",
        "text": "Merhaba herkese bugün proje toplantımızı yapıyoruz",
        "text_corrected": "Merhaba herkese bugün proje toplantımızı yapıyoruz",
        "inferred_speaker": "spk_0",
        "chunk": 0,
        "extracted_names": [],
        "embedding": null,
        "embedding_disabled": true,
        "mentioned_attendees": [],
        "candidate_speakers": [],
        "source": "raw_segment_fallback"
      },
      {
        "start": 5.2,
        "end": 12.8,
        "speaker": "spk_1",
        "text": "Öncelikle bütçe konusunu görüşmemiz gerekiyor",
        "text_corrected": "Öncelikle bütçe konusunu görüşmemiz gerekiyor",
        "inferred_speaker": "spk_1",
        "chunk": 0,
        "extracted_names": [],
        "embedding": null,
        "embedding_disabled": true,
        "mentioned_attendees": [],
        "candidate_speakers": [],
        "source": "raw_segment_fallback"
      },
      {
        "start": 12.8,
        "end": 20.5,
        "speaker": "spk_0",
        "text": "Tamam ben Q2 için finansal raporu hazırladım",
        "text_corrected": "Tamam ben Q2 için finansal raporu hazırladım",
        "inferred_speaker": "spk_0",
        "chunk": 0,
        "extracted_names": [],
        "embedding": null,
        "embedding_disabled": true,
        "mentioned_attendees": [],
        "candidate_speakers": [],
        "source": "raw_segment_fallback"
      }
    ],
    "path": "C:\\path\\to\\transcript_V7.json",
    "jsonl_path": "C:\\path\\to\\transcript.jsonl",
    "notification_result": {
      "status": "success",
      "message": "Transcript received",
      "meeting_id": "meeting-456"
    },
    "auto_analysis": {
      "prompt": "Summarize the key points discussed in this meeting",
      "answer": "Bu toplantıda proje ile ilgili önemli konular görüşüldü. Öncelikli olarak Q2 bütçe planlaması ele alındı ve finansal raporun hazırlandığı belirtildi. Ekip üyeleri arasında görev dağılımı yapıldı ve gelecek sprint için timeline belirlendi. Ayrıca müşteri geri bildirimleri değerlendirildi ve önümüzdeki hafta için aksiyon planı oluşturuldu.",
      "search_type": "hybrid",
      "processing_time_ms": 2340,
      "context_chunks": [
        {
          "text": "Öncelikle bütçe konusunu görüşmemiz gerekiyor",
          "speaker_name": "spk_1",
          "start_ms": 5200,
          "end_ms": 12800,
          "meeting_id": "meeting-456",
          "score": 0.95
        },
        {
          "text": "Tamam ben Q2 için finansal raporu hazırladım",
          "speaker_name": "spk_0",
          "start_ms": 12800,
          "end_ms": 20500,
          "meeting_id": "meeting-456",
          "score": 0.92
        },
        {
          "text": "Sprint planlaması için gelecek hafta toplanmalıyız",
          "speaker_name": "spk_1",
          "start_ms": 45000,
          "end_ms": 50000,
          "meeting_id": "meeting-456",
          "score": 0.88
        },
        {
          "text": "Müşteri geri bildirimleri çok pozitif geldi",
          "speaker_name": "spk_0",
          "start_ms": 120000,
          "end_ms": 125000,
          "meeting_id": "meeting-456",
          "score": 0.85
        },
        {
          "text": "Gelecek hafta için aksiyon planını hazırlayalım",
          "speaker_name": "spk_1",
          "start_ms": 180000,
          "end_ms": 185000,
          "meeting_id": "meeting-456",
          "score": 0.82
        }
      ],
      "chunk_count": 5
    }
  }
}
```

### Scenario 2: RAG Analysis Timeout (Fallback Case)

RAG servisi 30 saniye içinde yanıt vermezse veya hata olursa:

```json
{
  "json_output": {
    "status": "success",
    "session_id": "xyz789abc123",
    "segments": [
      {
        "start": 0.0,
        "end": 5.2,
        "speaker": "spk_0",
        "text": "Merhaba herkese",
        "text_corrected": "Merhaba herkese",
        "inferred_speaker": "spk_0",
        "chunk": 0
      }
    ],
    "path": "C:\\path\\to\\transcript_V7.json",
    "jsonl_path": "C:\\path\\to\\transcript.jsonl",
    "notification_result": {
      "status": "success",
      "message": "Transcript received",
      "meeting_id": "meeting-789"
    },
    "auto_analysis": null
  }
}
```

## Frontend Usage Example

### TypeScript Interface

```typescript
interface ContextChunk {
  text: string;
  speaker_name: string;
  start_ms: number;
  end_ms: number;
  meeting_id: string;
  score: number;
}

interface AutoAnalysis {
  prompt: string;
  answer: string;
  search_type: string;
  processing_time_ms: number;
  context_chunks: ContextChunk[];
  chunk_count: number;
}

interface TranscriptSegment {
  start: number;
  end: number;
  speaker: string;
  text: string;
  text_corrected: string;
  inferred_speaker: string;
  chunk: number;
  extracted_names?: string[];
  embedding?: number[] | null;
  embedding_disabled?: boolean;
  mentioned_attendees?: string[];
  candidate_speakers?: string[];
  source?: string;
}

interface TranscribeResponse {
  json_output: {
    status: string;
    session_id: string;
    segments: TranscriptSegment[];
    path: string;
    jsonl_path: string;
    notification_result?: any;
    auto_analysis: AutoAnalysis | null;
  };
}
```

### React Component Example

```typescript
import React, { useState } from 'react';

function TranscriptViewer() {
  const [transcriptData, setTranscriptData] = useState<TranscribeResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchTranscript = async (meetingId: string, videoPath: string) => {
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'current-user',
          video_path: videoPath,
          attendees: ['Speaker 1', 'Speaker 2'],
          meeting_id: meetingId
        })
      });

      const data: TranscribeResponse = await response.json();
      setTranscriptData(data);

    } catch (error) {
      console.error('Failed to fetch transcript:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div>Loading transcript...</div>;
  if (!transcriptData) return <div>No data</div>;

  const { segments, auto_analysis } = transcriptData.json_output;

  return (
    <div className="transcript-viewer">
      <h2>Meeting Transcript</h2>

      {/* Auto Analysis Section */}
      {auto_analysis ? (
        <div className="auto-analysis">
          <h3>AI Summary</h3>
          <p><strong>Prompt:</strong> {auto_analysis.prompt}</p>
          <p><strong>Answer:</strong> {auto_analysis.answer}</p>

          <div className="context-chunks">
            <h4>Key Moments ({auto_analysis.chunk_count})</h4>
            {auto_analysis.context_chunks.map((chunk, idx) => (
              <div key={idx} className="chunk">
                <span className="speaker">{chunk.speaker_name}</span>
                <span className="time">
                  {formatTime(chunk.start_ms)} - {formatTime(chunk.end_ms)}
                </span>
                <p>{chunk.text}</p>
                <span className="score">Relevance: {(chunk.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>

          <p className="processing-time">
            Processing time: {auto_analysis.processing_time_ms}ms
          </p>
        </div>
      ) : (
        <div className="auto-analysis-unavailable">
          <p>AI summary is not available for this meeting.</p>
        </div>
      )}

      {/* Transcript Segments */}
      <div className="segments">
        <h3>Full Transcript</h3>
        {segments.map((segment, idx) => (
          <div key={idx} className="segment">
            <span className="speaker">{segment.speaker}</span>
            <span className="time">
              {formatTime(segment.start * 1000)} - {formatTime(segment.end * 1000)}
            </span>
            <p>{segment.text_corrected || segment.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

export default TranscriptViewer;
```

## API Endpoints

### 1. POST /transcribe
**Request:**
```json
{
  "user_id": "user-123",
  "video_path": "https://example.com/video.mp4",
  "attendees": ["Speaker 1", "Speaker 2"],
  "meeting_id": "meeting-456"
}
```

**Response:** See "Complete Response Example" above

**Timing:**
- Waits up to 30 seconds for RAG analysis
- Returns immediately after pipeline completes if RAG already available
- Returns with `auto_analysis: null` if timeout

### 2. POST /rag/notify (Internal - Called by RAG Service)
**Request:**
```json
{
  "type": "auto_analysis_ready",
  "notification_id": "uuid-string",
  "org_id": "org-123",
  "meeting_id": "meeting-456",
  "generated_at": "2025-10-24T10:30:00.000Z",
  "auto_analysis": {
    "prompt": "Summarize the key points discussed in this meeting",
    "answer": "The generated LLM response...",
    "search_type": "hybrid",
    "processing_time_ms": 1234,
    "context_chunks": [...],
    "chunk_count": 5
  }
}
```

**Response:**
```json
{
  "status": "success",
  "meeting_id": "meeting-456",
  "message": "Auto analysis received and cached"
}
```

## Notes

1. **Timeout Handling**: Frontend should always check if `auto_analysis` is `null` before rendering
2. **Fallback UI**: Show a message like "AI summary unavailable" when `auto_analysis` is null
3. **Retry Logic**: If `auto_analysis` is null, frontend could optionally poll later or refresh
4. **Error Handling**: Standard HTTP error codes apply (500 for pipeline errors, 503 if dependencies unavailable)
5. **Cache Cleanup**: The service automatically cleans up cache entries after each request to prevent memory leaks

## Testing

Use the provided test script:
```bash
python test_rag_integration.py
```

This will:
1. Send a transcribe request
2. Simulate RAG notification after 2 seconds
3. Display the combined response
