import requests

# Replace with your access token
ACCESS_TOKEN = "EAAVuW66gWaEBOZBoRHpRWss3f7PFwfkp7QuGBDRgak7epQ8t7jRskSXJTAE8yNQXh3uN7UZCGIu8HIMyCdA653Qp2hPwkCqPeDDc1cyVlFqOn13VvlzV718jRZCmRogxtmipbdMPSU9Tcr85QFQB9tR4qD33AZAhxuihZCOkvEB0A1b3tbribubaopL4SDuL05CehbLdk35Q8aX2cFyILSkPVGCcA6lGEqXUY3eV4JsjWcGKEDgQFF7NZCQfp3snedeh3m0gZDZD"

# The Facebook username or profile link (you can use the user ID here)
PROFILE_LINK = "100088800420684"  # Use the user ID or username

# Define the Graph API endpoint
ENDPOINT = f'https://graph.facebook.com/v10.0/{PROFILE_LINK}?fields=id,name,picture&access_token={ACCESS_TOKEN}'

# Make the API request
response = requests.get(ENDPOINT)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print(f"User ID: {data['id']}")
    print(f"Name: {data['name']}")
    print(f"Profile Picture: {data['picture']['data']['url']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
