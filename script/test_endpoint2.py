import requests

def check_runpod_api_key(api_key: str) -> bool:
    """
    Checks if the provided RunPod API key is valid.
    Returns True if valid, False otherwise.
    """
    if not api_key or not isinstance(api_key, str):
        print("Error: API key must be a non-empty string.")
        return False

    url = "https://api.runpod.io/v2"  # Base endpoint
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }

    try:
        # A simple request to list endpoints (requires valid key)
        response = requests.get(f"{url}/endpoints", headers=headers, timeout=10)

        if response.status_code == 200:
            print("✅ API key is valid.")
            return True
        elif response.status_code == 401:
            print("❌ Invalid API key.")
            return False
        else:
            print(f"⚠ Unexpected response: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Network or request error: {e}")
        return False


# Example usage:
if __name__ == "__main__":
    # Replace with your actual RunPod API key
    my_api_key = ""
    check_runpod_api_key(my_api_key)
