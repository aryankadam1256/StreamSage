"""
Test TMDB API with different connection methods to diagnose the issue.
"""
import requests
import urllib3
import ssl

API_KEY = "626d6c744ce54f356ec6ce2d0ff3b6e6"
BASE_URL = "https://api.themoviedb.org/3"

print("="*70)
print("TMDB API CONNECTION DIAGNOSTICS")
print("="*70)

# Test 1: Standard request
print("\n[Test 1] Standard requests.get()...")
try:
    response = requests.get(
        f"{BASE_URL}/movie/550",
        params={"api_key": API_KEY},
        timeout=10
    )
    print(f"✓ SUCCESS! Status: {response.status_code}")
    print(f"  Movie: {response.json().get('title')}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 2: Disable SSL verification (not recommended for production)
print("\n[Test 2] With SSL verification disabled...")
try:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    response = requests.get(
        f"{BASE_URL}/movie/550",
        params={"api_key": API_KEY},
        timeout=10,
        verify=False
    )
    print(f"✓ SUCCESS! Status: {response.status_code}")
    print(f"  Movie: {response.json().get('title')}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 3: With custom headers (mimic browser)
print("\n[Test 3] With browser-like headers...")
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    response = requests.get(
        f"{BASE_URL}/movie/550",
        params={"api_key": API_KEY},
        headers=headers,
        timeout=10
    )
    print(f"✓ SUCCESS! Status: {response.status_code}")
    print(f"  Movie: {response.json().get('title')}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 4: With increased timeout
print("\n[Test 4] With 30 second timeout...")
try:
    response = requests.get(
        f"{BASE_URL}/movie/550",
        params={"api_key": API_KEY},
        timeout=30
    )
    print(f"✓ SUCCESS! Status: {response.status_code}")
    print(f"  Movie: {response.json().get('title')}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {str(e)[:100]}")

# Test 5: Using urllib instead of requests
print("\n[Test 5] Using urllib.request...")
try:
    import urllib.request
    import json
    url = f"{BASE_URL}/movie/550?api_key={API_KEY}"
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read())
        print(f"✓ SUCCESS! Status: {response.status}")
        print(f"  Movie: {data.get('title')}")
except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
