#!/usr/bin/env python3
"""
Test script to verify the notification functionality.
This script creates fake transcript data and tests the notification system.
"""

import json
import os
import tempfile
from pathlib import Path

def create_fake_transcript():
    """Create a fake transcript file for testing."""
    
    # Sample transcript data
    fake_transcript = [
        {
            "start": 0.0,
            "end": 5.2,
            "speaker": "Speaker1",
            "text": "Hello everyone, welcome to today's meeting."
        },
        {
            "start": 5.2,
            "end": 10.8,
            "speaker": "Speaker2", 
            "text": "Thank you for having me. I'm excited to discuss our project."
        },
        {
            "start": 10.8,
            "end": 15.5,
            "speaker": "Speaker1",
            "text": "Let's start with the quarterly review. How are we doing?"
        },
        {
            "start": 15.5,
            "end": 22.1,
            "speaker": "Speaker2",
            "text": "We've made excellent progress this quarter. Revenue is up 15%."
        },
        {
            "start": 22.1,
            "end": 28.3,
            "speaker": "Speaker1",
            "text": "That's fantastic news! What about the new product launch?"
        },
        {
            "start": 28.3,
            "end": 35.7,
            "speaker": "Speaker2",
            "text": "The launch is scheduled for next month. All systems are ready."
        }
    ]
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
    
    # Write transcript as JSONL
    for item in fake_transcript:
        temp_file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    temp_file.close()
    return temp_file.name

def test_notification_directly():
    """Test the notification function directly."""
    
    print("🧪 Testing Notification Function Directly")
    print("=" * 50)
    
    # Create fake transcript
    transcript_path = create_fake_transcript()
    print(f"✅ Created fake transcript: {transcript_path}")
    
    # Count lines and show content
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Transcript has {len(lines)} lines")
    print("📄 Sample content:")
    for i, line in enumerate(lines[:3]):  # Show first 3 lines
        data = json.loads(line)
        print(f"  {i+1}. [{data['start']:.1f}s-{data['end']:.1f}s] {data['speaker']}: {data['text']}")
    
    try:
        # Import and test notification function
        from app.pipeline.notification import notify_chunking
        
        print(f"\n🚀 Sending notification to chunking service...")
        print(f"   Meeting ID: test-meeting-123")
        print(f"   Transcript: {transcript_path}")
        print(f"   Chunking URL: http://localhost:8090")
        
        # Test notification
        result = notify_chunking(
            meeting_id="test-meeting-123",
            transcript_path=transcript_path,
            chunking_url="http://localhost:8090"
        )
        
        print(f"✅ Notification sent successfully!")
        print(f"📋 Response: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"❌ Notification failed: {e}")
        print(f"💡 This is expected if chunking service is not running on port 8090")
        
    finally:
        # Clean up
        if os.path.exists(transcript_path):
            os.unlink(transcript_path)
            print(f"🧹 Cleaned up test file: {transcript_path}")

def test_notification_with_mock_server():
    """Test notification with a mock chunking service."""
    
    print("\n" + "=" * 50)
    print("🌐 Testing with Mock Chunking Service")
    print("=" * 50)
    
    try:
        import httpx
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn
        import threading
        import time
        
        # Create mock chunking service
        mock_app = FastAPI()
        received_notifications = []
        
        @mock_app.post("/meetings/notify")
        async def mock_notify(request: dict):
            received_notifications.append(request)
            print(f"📨 Mock service received notification: {json.dumps(request, indent=2)}")
            return {"status": "success", "message": "Notification received"}
        
        # Start mock server in background
        def run_mock_server():
            uvicorn.run(mock_app, host="127.0.0.1", port=8090, log_level="error")
        
        server_thread = threading.Thread(target=run_mock_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Create fake transcript
        transcript_path = create_fake_transcript()
        print(f"✅ Created fake transcript: {transcript_path}")
        
        # Test notification
        from app.pipeline.notification import notify_chunking
        
        print(f"🚀 Sending notification to mock chunking service...")
        result = notify_chunking(
            meeting_id="test-meeting-456",
            transcript_path=transcript_path,
            chunking_url="http://127.0.0.1:8090"
        )
        
        print(f"✅ Notification sent successfully!")
        print(f"📋 Response: {json.dumps(result, indent=2)}")
        
        # Check if notification was received
        time.sleep(1)
        if received_notifications:
            print(f"🎉 Mock service received {len(received_notifications)} notification(s)")
            for i, notif in enumerate(received_notifications):
                print(f"   Notification {i+1}: {notif['meeting_id']} - {notif['count']} lines")
        else:
            print("⚠️  Mock service did not receive notification")
        
        # Clean up
        if os.path.exists(transcript_path):
            os.unlink(transcript_path)
            print(f"🧹 Cleaned up test file: {transcript_path}")
            
    except ImportError as e:
        print(f"❌ Missing dependency for mock test: {e}")
        print("💡 Install fastapi and uvicorn to run mock server test")
    except Exception as e:
        print(f"❌ Mock test failed: {e}")

def test_transcript_service_integration():
    """Test the full transcript service with notification."""
    
    print("\n" + "=" * 50)
    print("🔄 Testing Full Service Integration")
    print("=" * 50)
    
    try:
        import requests
        
        # Test data
        test_payload = {
            "user_id": "test_user",
            "video_path": "https://api-lia.arya-ai.com/api/public/meetings/1/af8f6828-6f9c-4331-afb4-8a3ae40349de_20250827_081821.mp4",
            "attendees": ["TestSpeaker1", "TestSpeaker2"],
            "meeting_id": "integration-test-789"
        }
        
        print(f"🚀 Sending request to transcript service...")
        print(f"   URL: http://localhost:8000/transcribe")
        print(f"   Payload: {json.dumps(test_payload, indent=2)}")
        
        # Send request
        response = requests.post(
            "http://localhost:8000/transcribe",
            json=test_payload,
            timeout=30  # 30 seconds timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Transcript service completed successfully!")
            print(f"📊 Status: {result['json_output']['status']}")
            print(f"🆔 Session ID: {result['json_output']['session_id']}")
            print(f"📄 Segments: {len(result['json_output']['segments'])}")
            
            if result['json_output'].get('notification_result'):
                print(f"📨 Notification sent: ✅")
                print(f"   Response: {result['json_output']['notification_result']}")
            else:
                print(f"📨 Notification: ❌ (no result)")
                
        else:
            print(f"❌ Service error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to transcript service")
        print("💡 Make sure the service is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

if __name__ == "__main__":
    print("🧪 Notification Function Test Suite")
    print("=" * 50)
    
    # Test 1: Direct notification function test
    test_notification_directly()
    
    # Test 2: Mock server test (optional)
    test_notification_with_mock_server()
    
    # Test 3: Full service integration test (optional)
    test_transcript_service_integration()
    
    print("\n" + "=" * 50)
    print("✅ Test suite completed!")
    print("\n💡 Notes:")
    print("- Direct notification test will fail if chunking service is not running")
    print("- Mock server test requires fastapi and uvicorn")
    print("- Integration test requires transcript service running on port 8000")
