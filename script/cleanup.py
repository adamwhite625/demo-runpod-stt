import requests
import toml
import os

# Lấy API Key từ secrets.toml hoặc biến môi trường
secrets_path = '.streamlit/secrets.toml'
api_key = os.environ.get("RUNPOD_API_KEY")

if not api_key and os.path.exists(secrets_path):
    secrets = toml.load(secrets_path)
    api_key = secrets.get('runpod', {}).get('api_key')

if not api_key:
    print("Lỗi: Không tìm thấy API Key.")
    exit(1)

headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}

def delete_endpoint(ep_id):
    mutation = '''
    mutation deleteEndpoint($id: String!) { deleteEndpoint(id: $id) }
    '''
    resp = requests.post('https://api.runpod.io/graphql', json={'query': mutation, 'variables': {'id': ep_id}}, headers=headers)
    return resp.json()

# ID của endpoint bạn muốn xóa
TARGET_ENDPOINT = "gsl3zucliyq5wz"

print("-" * 50)
print(f"Đang tiến hành xóa trực tiếp endpoint: {TARGET_ENDPOINT}...")
print("-" * 50)

result = delete_endpoint(TARGET_ENDPOINT)

# Kiểm tra kết quả trả về
if 'errors' in result:
    print("Xóa thất bại. Chi tiết lỗi:")
    print(result['errors'])
else:
    print(f"Đã gửi lệnh xóa thành công cho endpoint {TARGET_ENDPOINT}!")
    print("Kết quả trả về từ server:", result)