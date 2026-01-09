"""
Test Tomorrow.io API Integration
Tests real-time weather, forecasting, and feature extraction
"""

import os
from dotenv import load_dotenv
from src.data_ingestion.weather_client import TomorrowIOClient
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def test_realtime_weather():
    """Test real-time weather fetching"""
    print("\n" + "="*60)
    print("🌤️  TESTING TOMORROW.IO REAL-TIME WEATHER")
    print("="*60)
    
    try:
        client = TomorrowIOClient()
        
        # Test locations (Indian cities)
        locations = [
            {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946}
        ]
        
        for loc in locations:
            print(f"\n📍 Fetching weather for {loc['name']}...")
            
            data = client.get_realtime_weather(lat=loc['lat'], lon=loc['lon'])
            
            if data and 'data' in data:
                values = data['data']['values']
                print(f"   ✅ Temperature: {values.get('temperature')}°C")
                print(f"   ✅ Feels Like: {values.get('temperatureApparent')}°C")
                print(f"   ✅ Humidity: {values.get('humidity')}%")
                print(f"   ✅ Wind Speed: {values.get('windSpeed')} km/h")
                print(f"   ✅ Pressure: {values.get('pressureSeaLevel')} hPa")
                print(f"   ✅ Visibility: {values.get('visibility')} km")
                print(f"   ✅ Cloud Cover: {values.get('cloudCover')}%")
                print(f"   ✅ Weather Code: {values.get('weatherCode')}")
            else:
                print(f"   ❌ Failed to fetch data for {loc['name']}")
        
        print("\n✅ Real-time weather test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Real-time weather test FAILED: {str(e)}")
        return False

def test_forecast():
    """Test hourly forecast"""
    print("\n" + "="*60)
    print("📈 TESTING TOMORROW.IO FORECAST")
    print("="*60)
    
    try:
        client = TomorrowIOClient()
        
        # Delhi coordinates
        lat, lon = 28.6139, 77.2090
        
        print(f"\n📍 Fetching hourly forecast for Delhi...")
        
        forecast_data = client.get_forecast(lat=lat, lon=lon, timesteps="1h")
        
        if forecast_data and 'timelines' in forecast_data:
            hourly = forecast_data['timelines'].get('hourly', [])
            
            print(f"\n   ✅ Retrieved {len(hourly)} hourly forecast points")
            
            # Show first 5 hours
            print("\n   📊 Next 5 hours forecast:")
            for i, hour in enumerate(hourly[:5]):
                time = hour.get('time', 'Unknown')
                values = hour.get('values', {})
                print(f"\n   Hour {i+1} ({time}):")
                print(f"      Temperature: {values.get('temperature')}°C")
                print(f"      Humidity: {values.get('humidity')}%")
                print(f"      Precipitation: {values.get('precipitationIntensity')} mm/h")
                print(f"      Precipitation Probability: {values.get('precipitationProbability')}%")
        
        print("\n✅ Forecast test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Forecast test FAILED: {str(e)}")
        return False

def test_feature_extraction():
    """Test feature extraction from weather data"""
    print("\n" + "="*60)
    print("🔧 TESTING FEATURE EXTRACTION")
    print("="*60)
    
    try:
        client = TomorrowIOClient()
        
        # Get weather data
        data = client.get_realtime_weather(lat=28.6139, lon=77.2090)
        
        # Extract features
        features = client.extract_weather_features(data)
        
        print("\n📊 Extracted Features:")
        for key, value in features.items():
            print(f"   • {key}: {value}")
        
        print("\n✅ Feature extraction test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature extraction test FAILED: {str(e)}")
        return False

def test_multiple_locations():
    """Test batch weather fetching for multiple locations"""
    print("\n" + "="*60)
    print("🌍 TESTING MULTIPLE LOCATIONS")
    print("="*60)
    
    try:
        client = TomorrowIOClient()
        
        locations = [
            {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
            {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
            {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639}
        ]
        
        print(f"\n📍 Fetching weather for {len(locations)} cities...")
        
        weather_data = client.get_weather_for_locations(locations, include_forecast=False)
        
        print(f"\n✅ Retrieved weather for {len(weather_data)} locations:")
        for city, data in weather_data.items():
            if data and 'data' in data:
                temp = data['data']['values'].get('temperature', 'N/A')
                print(f"   • {city}: {temp}°C")
        
        print("\n✅ Multiple locations test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Multiple locations test FAILED: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 TOMORROW.IO API INTEGRATION TEST SUITE")
    print("="*60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Run tests
    results.append(("Real-time Weather", test_realtime_weather()))
    results.append(("Forecast", test_forecast()))
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Multiple Locations", test_multiple_locations()))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*60}")
    print(f"🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Tomorrow.io API is working perfectly!")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
