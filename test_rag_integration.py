"""
Test script for RAG auto_analysis integration

This script demonstrates how to:
1. Call POST /transcribe endpoint
2. Call POST /rag/notify endpoint (simulating RAG service)
3. Verify the combined response

Usage:
    python test_rag_integration.py
"""

import requests
import json
import time
from threading import Thread

# Configuration
BASE_URL = "http://localhost:8000"
TRANSCRIBE_URL = f"{BASE_URL}/transcribe"
RAG_NOTIFY_URL = f"{BASE_URL}/rag/notify"

def test_rag_integration():
    """Test the full RAG integration flow"""

    # Test data
    meeting_id = "test-meeting-123"

    transcribe_payload = {
        "user_id": "test-user",
        "video_path": "https://example.com/test-video.mp4",  # Replace with real URL
        "attendees": ["Speaker 1", "Speaker 2"],
        "meeting_id": meeting_id
    }

    rag_notification_payload = {
        "type": "auto_analysis_ready",
        "notification_id": "test-notification-456",
        "org_id": "org-123",
        "meeting_id": meeting_id,
        "generated_at": "2025-10-24T10:30:00.000Z",
        "auto_analysis": {
            "prompt": "Summarize the key points discussed in this meeting",
            "answer": "The meeting covered several important topics including project timeline, budget allocation, and team responsibilities. Key decisions were made regarding the next sprint planning.",
            "search_type": "hybrid",
            "processing_time_ms": 1234,
            "context_chunks": [
                {
                    "text": "We need to finalize the project timeline by next week",
                    "speaker_name": "Speaker 1",
                    "start_ms": 1000,
                    "end_ms": 5000,
                    "meeting_id": meeting_id,
                    "score": 0.95
                },
                {
                    "text": "Budget allocation for Q2 needs approval",
                    "speaker_name": "Speaker 2",
                    "start_ms": 6000,
                    "end_ms": 10000,
                    "meeting_id": meeting_id,
                    "score": 0.87
                }
            ],
            "chunk_count": 2
        }
    }

    print("=" * 60)
    print("🧪 Testing RAG Auto-Analysis Integration")
    print("=" * 60)

    # Simulate RAG notification arriving after 2 seconds
    def send_rag_notification_delayed():
        print("\n⏰ Waiting 2 seconds before sending RAG notification...")
        time.sleep(2)

        print(f"\n📤 Sending RAG notification to: {RAG_NOTIFY_URL}")
        print(f"   Meeting ID: {meeting_id}")

        response = requests.post(RAG_NOTIFY_URL, json=rag_notification_payload)

        if response.status_code == 200:
            print(f"✅ RAG notification sent successfully")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ RAG notification failed: {response.status_code}")
            print(f"   Error: {response.text}")

    # Start background thread to send RAG notification
    rag_thread = Thread(target=send_rag_notification_delayed, daemon=True)
    rag_thread.start()

    # Call transcribe endpoint
    print(f"\n📤 Sending transcribe request to: {TRANSCRIBE_URL}")
    print(f"   Meeting ID: {meeting_id}")
    print(f"   Video path: {transcribe_payload['video_path']}")

    try:
        response = requests.post(TRANSCRIBE_URL, json=transcribe_payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print("\n✅ Transcribe request completed successfully!")
            print("\n" + "=" * 60)
            print("📊 RESPONSE STRUCTURE:")
            print("=" * 60)

            json_output = result.get("json_output", {})

            print(f"\n🔹 Status: {json_output.get('status')}")
            print(f"🔹 Session ID: {json_output.get('session_id')}")
            print(f"🔹 Segments count: {len(json_output.get('segments', []))}")

            # Check if auto_analysis is included
            if "auto_analysis" in json_output:
                auto_analysis = json_output["auto_analysis"]
                print("\n✅ AUTO ANALYSIS INCLUDED:")
                print(f"   - Prompt: {auto_analysis.get('prompt')}")
                print(f"   - Answer: {auto_analysis.get('answer')[:100]}...")
                print(f"   - Search type: {auto_analysis.get('search_type')}")
                print(f"   - Processing time: {auto_analysis.get('processing_time_ms')}ms")
                print(f"   - Chunk count: {auto_analysis.get('chunk_count')}")

                if auto_analysis.get('context_chunks'):
                    print(f"\n   Context chunks ({len(auto_analysis['context_chunks'])}):")
                    for i, chunk in enumerate(auto_analysis['context_chunks'][:2], 1):
                        print(f"     {i}. {chunk['text'][:50]}... (score: {chunk['score']})")
            else:
                print("\n⚠️  AUTO ANALYSIS NOT INCLUDED (timeout or not received)")

            print("\n" + "=" * 60)
            print("📄 Full Response (formatted):")
            print("=" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        else:
            print(f"\n❌ Transcribe request failed: {response.status_code}")
            print(f"   Error: {response.text}")

    except requests.exceptions.Timeout:
        print("\n⏰ Request timed out")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Wait for background thread
    rag_thread.join(timeout=5)

    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag_integration()
