"""
Tomorrow.io API Client for weather data and forecasting
"""

import os
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

from .base_client import BaseAPIClient


class TomorrowIOClient(BaseAPIClient):
    """Client for Tomorrow.io API with advanced weather and forecasting features"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 25,  # Free tier: 25 requests/hour, 500 requests/day
        timeout: int = 30,
        units: str = "metric"
    ):
        """
        Initialize Tomorrow.io client
        
        Args:
            api_key: Tomorrow.io API key
            rate_limit: Maximum requests per hour (free tier: 25/hour)
            timeout: Request timeout in seconds
            units: Units of measurement (metric, imperial)
        """
        if api_key is None:
            api_key = os.getenv("TOMORROW_IO_API_KEY")
        
        if not api_key:
            raise ValueError("Tomorrow.io API key is required")
        
        super().__init__(
            base_url="https://api.tomorrow.io/v4",
            api_key=api_key,
            rate_limit=rate_limit,
            timeout=timeout
        )
        
        self.units = units
        logger.info("Tomorrow.io API client initialized")
    
    def _format_location(self, lat: float, lon: float) -> str:
        """Format location as lat,lon string"""
        return f"{lat},{lon}"
    
    def get_realtime_weather(
        self,
        lat: float,
        lon: float,
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get real-time weather data
        
        Args:
            lat: Latitude
            lon: Longitude
            fields: List of weather fields to retrieve. If None, gets all common fields.
                   Available: temperature, temperatureApparent, humidity, windSpeed,
                   windDirection, pressureSeaLevel, precipitationIntensity, visibility,
                   cloudCover, weatherCode, etc.
            
        Returns:
            Real-time weather data dictionary
        """
        try:
            endpoint = "weather/realtime"
            
            # Default fields if none specified
            if fields is None:
                fields = [
                    "temperature",
                    "temperatureApparent",
                    "humidity",
                    "windSpeed",
                    "windDirection",
                    "pressureSeaLevel",
                    "precipitationIntensity",
                    "visibility",
                    "cloudCover",
                    "weatherCode"
                ]
            
            params = {
                "location": self._format_location(lat, lon),
                "apikey": self.api_key,
                "units": self.units
            }
            
            logger.info(f"Fetching real-time weather for ({lat}, {lon})")
            data = self.get(endpoint, params=params)
            
            # Add fetch timestamp
            data['fetched_at'] = datetime.now().isoformat()
            
            logger.info(f"Retrieved real-time weather data")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get real-time weather: {str(e)}")
            return {}
    
    def get_forecast(
        self,
        lat: float,
        lon: float,
        timesteps: str = "1h",
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get weather forecast (supports 1min, 5min, 15min, 30min, 1h, 1d intervals)
        
        Args:
            lat: Latitude
            lon: Longitude
            timesteps: Forecast interval - "1h" (hourly), "1d" (daily), etc.
            fields: List of weather fields to retrieve
            
        Returns:
            Forecast data dictionary with timelines
        """
        try:
            endpoint = "weather/forecast"
            
            # Default fields if none specified
            if fields is None:
                fields = [
                    "temperature",
                    "temperatureApparent",
                    "humidity",
                    "windSpeed",
                    "windDirection",
                    "pressureSeaLevel",
                    "precipitationIntensity",
                    "precipitationProbability",
                    "visibility",
                    "cloudCover",
                    "weatherCode"
                ]
            
            params = {
                "location": self._format_location(lat, lon),
                "apikey": self.api_key,
                "units": self.units,
                "timesteps": timesteps
            }
            
            logger.info(f"Fetching {timesteps} forecast for ({lat}, {lon})")
            data = self.get(endpoint, params=params)
            
            # Add fetch timestamp
            data['fetched_at'] = datetime.now().isoformat()
            
            num_forecasts = len(data.get('timelines', {}).get('hourly', [])) if timesteps == '1h' else len(data.get('timelines', {}).get('daily', []))
            logger.info(f"Retrieved {num_forecasts} forecast data points")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get weather forecast: {str(e)}")
            return {}
    
    def get_historical_weather(
        self,
        lat: float,
        lon: float,
        start_time: datetime,
        end_time: datetime,
        timesteps: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get historical weather data
        
        Args:
            lat: Latitude
            lon: Longitude
            start_time: Start datetime
            end_time: End datetime
            timesteps: Data interval
            
        Returns:
            Historical weather data
        """
        try:
            endpoint = "weather/history/recent"
            
            params = {
                "location": self._format_location(lat, lon),
                "apikey": self.api_key,
                "units": self.units,
                "timesteps": timesteps,
                "startTime": start_time.isoformat(),
                "endTime": end_time.isoformat()
            }
            
            logger.info(f"Fetching historical weather for ({lat}, {lon})")
            data = self.get(endpoint, params=params)
            
            data['fetched_at'] = datetime.now().isoformat()
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical weather: {str(e)}")
            return {}
    
    def extract_weather_features(self, weather_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract relevant features from Tomorrow.io weather data for ML model
        
        Args:
            weather_data: Raw weather data from API
            
        Returns:
            Dictionary of weather features
        """
        try:
            # Handle both realtime and forecast data structures
            if 'data' in weather_data:
                values = weather_data['data'].get('values', {})
            elif 'timelines' in weather_data:
                # Get first forecast point
                timelines = weather_data['timelines']
                if 'hourly' in timelines and timelines['hourly']:
                    values = timelines['hourly'][0].get('values', {})
                elif 'daily' in timelines and timelines['daily']:
                    values = timelines['daily'][0].get('values', {})
                else:
                    values = {}
            else:
                values = weather_data.get('values', {})
            
            features = {
                'temperature': values.get('temperature', 0),
                'feels_like': values.get('temperatureApparent', 0),
                'pressure': values.get('pressureSeaLevel', 1013),  # hPa
                'humidity': values.get('humidity', 0),  # %
                'wind_speed': values.get('windSpeed', 0),  # m/s or km/h based on units
                'wind_deg': values.get('windDirection', 0),  # degrees
                'cloudiness': values.get('cloudCover', 0),  # %
                'visibility': values.get('visibility', 10),  # km
                'precipitation_intensity': values.get('precipitationIntensity', 0),  # mm/hr
                'precipitation_probability': values.get('precipitationProbability', 0),  # %
                'weather_code': values.get('weatherCode', 0),  # Tomorrow.io weather code
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract weather features: {str(e)}")
            return {}
    
    def get_weather_for_locations(
        self,
        locations: List[Dict[str, Any]],
        include_forecast: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current weather for multiple locations
        
        Args:
            locations: List of location dictionaries with 'name', 'lat', 'lon'
            include_forecast: Whether to include forecast data
            
        Returns:
            Dictionary mapping location names to weather data
        """
        weather_data = {}
        
        for location in locations:
            name = location.get('name', 'Unknown')
            lat = location.get('lat')
            lon = location.get('lon')
            
            if lat is None or lon is None:
                logger.warning(f"Missing coordinates for {name}")
                continue
            
            try:
                data = self.get_realtime_weather(lat=lat, lon=lon)
                
                if include_forecast:
                    forecast = self.get_forecast(lat=lat, lon=lon, timesteps="1h")
                    data['forecast'] = forecast
                
                weather_data[name] = data
                
            except Exception as e:
                logger.error(f"Failed to get weather for {name}: {str(e)}")
                weather_data[name] = {}
        
        return weather_data
    
    def save_weather_data(
        self,
        weather_data: Dict[str, Any],
        output_path: Path,
        data_type: str = "realtime"
    ) -> None:
        """
        Save weather data to CSV file
        
        Args:
            weather_data: Weather data dictionary
            output_path: Output file path
            data_type: Type of data - "realtime", "forecast", or "historical"
        """
        try:
            records = []
            
            if data_type == "realtime":
                # Handle realtime data
                features = self.extract_weather_features(weather_data)
                features['timestamp'] = weather_data.get('data', {}).get('time', datetime.now().isoformat())
                records.append(features)
                
            elif data_type == "forecast":
                # Handle forecast data
                timelines = weather_data.get('timelines', {})
                for timeline_type in ['hourly', 'daily']:
                    if timeline_type in timelines:
                        for interval in timelines[timeline_type]:
                            features = self.extract_weather_features({'values': interval.get('values', {})})
                            features['timestamp'] = interval.get('time', '')
                            records.append(features)
            
            df = pd.DataFrame(records)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(records)} weather records to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save weather data: {str(e)}")
    
    def calculate_pressure_gradient(
        self,
        historical_pressure: List[float],
        time_interval_hours: float = 1.0
    ) -> float:
        """
        Calculate pressure gradient (rate of change)
        
        Args:
            historical_pressure: List of pressure values in chronological order
            time_interval_hours: Time interval between measurements
            
        Returns:
            Pressure gradient in hPa/hour
        """
        if len(historical_pressure) < 2:
            return 0.0
        
        # Calculate average gradient
        gradients = []
        for i in range(1, len(historical_pressure)):
            gradient = (historical_pressure[i] - historical_pressure[i-1]) / time_interval_hours
            gradients.append(gradient)
        
        avg_gradient = sum(gradients) / len(gradients) if gradients else 0.0
        return avg_gradient
    
    def get_weather_alerts(
        self,
        lat: float,
        lon: float
    ) -> List[Dict[str, Any]]:
        """
        Get weather alerts for a location (if available in API plan)
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            List of active weather alerts
        """
        try:
            # Note: Alerts may require higher tier plan
            endpoint = "weather/alerts"
            
            params = {
                "location": self._format_location(lat, lon),
                "apikey": self.api_key
            }
            
            logger.info(f"Fetching weather alerts for ({lat}, {lon})")
            data = self.get(endpoint, params=params)
            
            return data.get('alerts', [])
            
        except Exception as e:
            logger.warning(f"Weather alerts not available: {str(e)}")
            return []


# Convenience function
def create_weather_client() -> TomorrowIOClient:
    """Create and return a Tomorrow.io client instance"""
    return TomorrowIOClient()


# Backward compatibility alias
OpenWeatherClient = TomorrowIOClient
