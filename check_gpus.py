"""List all GPU types that currently have available serverless nodes."""
import requests
import toml

secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets['runpod']['api_key']
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

query = '''
query {
  gpuTypes {
    id
    displayName
    nodeGroupDatacenters { id }
    lowestPrice(input: { gpuCount: 1, secureCloud: true }) {
      uninterruptablePrice
    }
  }
}
'''
resp = requests.post('https://api.runpod.io/graphql', json={'query': query}, headers=headers)
gpus = resp.json().get('data', {}).get('gpuTypes', [])

print("Available GPU | Price/hr | Datacenters")
print("-" * 60)
for g in gpus:
    dcs = [d['id'] for d in g.get('nodeGroupDatacenters', [])]
    if dcs:
        price = g.get('lowestPrice', {}).get('uninterruptablePrice', 'N/A')
        print(f"  {g['displayName']}: ${price}/hr -> {dcs}")
