import requests

url = "http://127.0.0.1:8000/search"
data = {"query": "What is nursing care?"}

response = requests.post(url, json=data)
print("Status code:", response.status_code)
print("Response:", response.json())
