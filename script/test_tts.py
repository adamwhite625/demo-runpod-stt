"""Quick test script to debug TTS endpoint failure."""
import json
import requests
import toml

secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets['runpod']['api_key']
endpoint_id = secrets['runpod']['endpoint_id']

payload = {
    "input": {
        "action": "tts",
        "payload": {
            "text": "Xin chao",
            "language": "vi"
        }
    }
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

print("Submitting TTS job...")
resp = requests.post(f"https://api.runpod.ai/v2/{endpoint_id}/runsync", json=payload, headers=headers, timeout=300)
print("Response status:", resp.status_code)
print("Response JSON:")
print(json.dumps(resp.json(), indent=2))
