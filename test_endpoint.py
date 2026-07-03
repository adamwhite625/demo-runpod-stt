"""Quick test script to verify the RunPod endpoint is working."""
import base64
import json
import requests
import toml

secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets['runpod']['api_key']
endpoint_id = secrets['runpod']['endpoint_id']

with open('demo_stt_small.mp3', 'rb') as f:
    audio_b64 = base64.b64encode(f.read()).decode('utf-8')

# Nest data inside 'payload' key to match voice_engine(action, payload) signature
payload = {
    "input": {
        "action": "stt",
        "payload": {
            "audio_base64": audio_b64,
            "language": "vi"
        }
    }
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("Submitting job...")
resp = requests.post(f"https://api.runpod.ai/v2/{endpoint_id}/runsync", json=payload, headers=headers, timeout=300)
print("Response status:", resp.status_code)
print("Response JSON:")
print(json.dumps(resp.json(), indent=2))
