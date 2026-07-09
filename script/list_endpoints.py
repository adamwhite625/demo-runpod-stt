import requests
import toml

secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets['runpod']['api_key']

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

query = '''
query {
  myself {
    endpoints {
      id
      name
    }
  }
}
'''

resp = requests.post(
    'https://api.runpod.io/graphql',
    json={'query': query},
    headers=headers
)

print(resp.json())