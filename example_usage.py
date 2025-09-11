#!/usr/bin/env python3
"""
Example usage of the updated transcript service with notification.
"""

import requests
import json

def example_transcript_request():
    """Example of how to call the transcript service with notification."""
    
    # Service configuration
    service_url = "http://localhost:8000"
    
    # Request payload with new required fields
    payload = {
        "user_id": "example_user",
        "video_path": "https://api-lia.arya-ai.com/api/public/meetings/1/af8f6828-6f9c-4331-afb4-8a3ae40349de_20250827_081821.mp4",
        "attendees": ["Ufuk", "Samet"],
        "meeting_id": "meeting-123"
    }
    
    print("Example Transcript Service Request")
    print("=" * 40)
    print(f"Service URL: {service_url}")
    print(f"Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        # Make the request
        response = requests.post(
            f"{service_url}/transcribe",
            json=payload,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print("Response structure:")
            print(f"- Status: {result['json_output']['status']}")
            print(f"- Session ID: {result['json_output']['session_id']}")
            print(f"- Segments count: {len(result['json_output']['segments'])}")
            print(f"- JSON path: {result['json_output']['path']}")
            print(f"- JSONL path: {result['json_output']['jsonl_path']}")
            
            if result['json_output'].get('notification_result'):
                print(f"- Notification sent: ✅")
                print(f"  Chunking service response: {result['json_output']['notification_result']}")
            else:
                print(f"- Notification sent: ❌ (no result)")
                
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def example_curl_command():
    """Example curl command for testing the API."""
    
    print("\n" + "=" * 40)
    print("Example cURL command:")
    print("=" * 40)
    
    curl_command = '''curl -X POST "http://localhost:8000/transcribe" \\
  -H "Content-Type: application/json" \\
  -d '{
    "user_id": "example_user",
    "video_path": "https://api-lia.arya-ai.com/api/public/meetings/1/af8f6828-6f9c-4331-afb4-8a3ae40349de_20250827_081821.mp4",
    "attendees": ["Ufuk", "Samet"],
    "meeting_id": "meeting-123"
  }' '''
    
    print(curl_command)

if __name__ == "__main__":
    example_transcript_request()
    example_curl_command()
    
    print("\n" + "=" * 40)
    print("Notes:")
    print("- Make sure the transcript service is running on port 8000")
    print("- Make sure the chunking service is running on port 8090")
    print("- Update the video_path to point to a valid video file")
    print("- The service will automatically send a notification to the chunking service")
