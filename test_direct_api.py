"""
Quick test - Direct Tomorrow.io API call
"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = "sHS9mA2DaGh1KI6jfFxRuACLLDXG0aKg"

# Delhi coordinates
lat, lon = 28.6139, 77.2090

url = f"https://api.tomorrow.io/v4/weather/realtime"
params = {
    'location': f"{lat},{lon}",
    'apikey': API_KEY,
    'units': 'metric'
}

print(f"Testing Tomorrow.io API...")
print(f"URL: {url}")
print(f"Params: {params}")

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        data = response.json()
        values = data['data']['values']
        print(f"\n✅ SUCCESS!")
        print(f"Temperature: {values['temperature']}°C")
        print(f"Humidity: {values['humidity']}%")
        print(f"Wind Speed: {values['windSpeed']} km/h")
        print(f"Pressure: {values['pressureSeaLevel']} hPa")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
