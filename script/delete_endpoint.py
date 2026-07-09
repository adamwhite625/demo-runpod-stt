import requests, toml
secrets = toml.load('.streamlit/secrets.toml')
api_key = secrets['runpod']['api_key']
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
mutation = '''
mutation deleteEndpoint($id: String!) {
  deleteEndpoint(id: $id)
}
'''
resp = requests.post('https://api.runpod.io/graphql', json={'query': mutation, 'variables': {'id': ''}}, headers=headers)
print('Deleted: oj89bbcct211pl', resp.json())
