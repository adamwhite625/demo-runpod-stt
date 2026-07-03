"""Create a new Network Volume in US-KS-2 where A100 SXM GPUs are available."""
import requests
import toml

secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets['runpod']['api_key']
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

payload = {'dataCenterId': 'US-KS-2', 'name': 'demo-stt-models', 'size': 20}
resp = requests.post('https://rest.runpod.io/v1/networkvolumes', json=payload, headers=headers)
print(resp.status_code, resp.json())
