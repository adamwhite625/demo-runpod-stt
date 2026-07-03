"""Script to create RunPod Network Volume and list available datacenters."""

import requests
import sys
import os
import toml

def get_api_key():
    # Thử đọc từ secrets.toml trước
    secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r") as f:
                secrets = toml.load(f)
                return secrets.get("runpod", {}).get("api_key", "")
        except Exception:
            pass
    return os.environ.get("RUNPOD_API_KEY", "")

API_KEY = get_api_key()
BASE_URL = "https://rest.runpod.io/v1"

def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

def list_volumes():
    """List all existing network volumes."""
    resp = requests.get(f"{BASE_URL}/networkvolumes", headers=get_headers())
    resp.raise_for_status()
    volumes = resp.json()
    if not volumes:
        print("No network volumes found.")
        return
    print("\n=== Existing Network Volumes ===")
    for v in volumes:
        print(f"  ID: {v.get('id')}  |  Name: {v.get('name')}  |  Size: {v.get('size')}GB  |  DC: {v.get('dataCenterId')}")

def create_volume(name: str, size: int, datacenter_id: str):
    """Create a new network volume."""
    payload = {
        "dataCenterId": datacenter_id,
        "name": name,
        "size": size,
    }
    resp = requests.post(f"{BASE_URL}/networkvolumes", json=payload, headers=get_headers())
    resp.raise_for_status()
    data = resp.json()
    print(f"\nVolume created successfully!")
    print(f"  ID: {data.get('id')}")
    print(f"  Name: {data.get('name')}")
    print(f"  Size: {data.get('size')}GB")
    print(f"  Datacenter: {data.get('dataCenterId')}")
    return data.get("id")

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Không tìm thấy API Key.")
        print("Vui lòng đảm bảo bạn đã cấu hình api_key trong file .streamlit/secrets.toml")
        sys.exit(1)
    
    print("Checking existing volumes...")
    list_volumes()
    
    print("\n--- Creating new volume for AI models ---")
    volume_id = create_volume(
        name="voice-bot-models",
        size=20,
        datacenter_id="US-TX-3"  # Change this to match your endpoint's datacenter
    )
    print(f"\nDone! Use this volume ID in handler.py: {volume_id}")
