
import requests
import json

URL = "http://localhost:8001/ask"
payload = {
    "query": "What happens in the ending of the movie?",
    "movie_id": "interstellar",
    "top_k": 3
}

print(f"Testing Oracle service at {URL}...")
try:
    response = requests.post(URL, json=payload, timeout=60)
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")

