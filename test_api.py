#!/usr/bin/env python3
import requests
import json

# Test the backend API directly
print("=== Testing Backend API ===")
try:
    response = requests.get("http://localhost:8000/docs")
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal documents returned: {len(data)}")
        for doc in data:
            print(f"\n  Name: {doc.get('name')}")
            print(f"  Hash: {doc.get('hash', 'MISSING')[:16] if doc.get('hash') else 'MISSING'}...")
            print(f"  Status: {doc.get('status')}")
            print(f"  Size: {doc.get('size')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")

# Test through frontend proxy
print("\n\n=== Testing Frontend Proxy ===")
try:
    response = requests.get("http://localhost:5173/api/docs")
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal documents returned: {len(data)}")
        for doc in data:
            print(f"\n  Name: {doc.get('name')}")
            print(f"  Hash: {doc.get('hash', 'MISSING')[:16] if doc.get('hash') else 'MISSING'}...")
            print(f"  Status: {doc.get('status')}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {e}")
