"""
Environmental feature engineering and stress index calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from loguru import logger


class EnvironmentalFeatureExtractor:
    """Extract and engineer environmental features for stress prediction"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize environmental feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        logger.info("Initialized EnvironmentalFeatureExtractor")
    
    def normalize_temperature(
        self,
        temperature: float,
        min_temp: float = -10.0,
        max_temp: float = 50.0
    ) -> float:
        """
        Normalize temperature to [0, 1] range
        
        Args:
            temperature: Temperature in Celsius
            min_temp: Minimum expected temperature
            max_temp: Maximum expected temperature
            
        Returns:
            Normalized temperature
        """
        return (temperature - min_temp) / (max_temp - min_temp)
    
    def normalize_pressure(
        self,
        pressure: float,
        min_pressure: float = 950.0,
        max_pressure: float = 1050.0
    ) -> float:
        """
        Normalize atmospheric pressure to [0, 1] range
        
        Args:
            pressure: Pressure in hPa
            min_pressure: Minimum expected pressure
            max_pressure: Maximum expected pressure
            
        Returns:
            Normalized pressure
        """
        return (pressure - min_pressure) / (max_pressure - min_pressure)
    
    def normalize_humidity(self, humidity: float) -> float:
        """
        Normalize humidity to [0, 1] range
        
        Args:
            humidity: Humidity percentage
            
        Returns:
            Normalized humidity
        """
        return humidity / 100.0
    
    def calculate_pressure_gradient(
        self,
        pressure_history: List[float],
        time_intervals: List[float]
    ) -> float:
        """
        Calculate rate of pressure change
        
        Args:
            pressure_history: List of pressure values (hPa)
            time_intervals: List of time intervals (hours)
            
        Returns:
            Average pressure gradient (hPa/hour)
        """
        if len(pressure_history) < 2:
            return 0.0
        
        gradients = []
        for i in range(1, len(pressure_history)):
            dt = time_intervals[i] - time_intervals[i-1]
            if dt > 0:
                gradient = (pressure_history[i] - pressure_history[i-1]) / dt
                gradients.append(gradient)
        
        return np.mean(gradients) if gradients else 0.0
    
    def calculate_temperature_gradient(
        self,
        temperature_history: List[float],
        time_intervals: List[float]
    ) -> float:
        """
        Calculate rate of temperature change
        
        Args:
            temperature_history: List of temperature values (Celsius)
            time_intervals: List of time intervals (hours)
            
        Returns:
            Average temperature gradient (°C/hour)
        """
        if len(temperature_history) < 2:
            return 0.0
        
        gradients = []
        for i in range(1, len(temperature_history)):
            dt = time_intervals[i] - time_intervals[i-1]
            if dt > 0:
                gradient = (temperature_history[i] - temperature_history[i-1]) / dt
                gradients.append(gradient)
        
        return np.mean(gradients) if gradients else 0.0
    
    def calculate_wind_chill(
        self,
        temperature: float,
        wind_speed: float
    ) -> float:
        """
        Calculate wind chill factor
        
        Args:
            temperature: Temperature in Celsius
            wind_speed: Wind speed in m/s
            
        Returns:
            Wind chill temperature
        """
        # Convert m/s to km/h
        wind_speed_kmh = wind_speed * 3.6
        
        # Wind chill formula (valid for temp <= 10°C and wind >= 4.8 km/h)
        if temperature <= 10 and wind_speed_kmh >= 4.8:
            wind_chill = (
                13.12 + 0.6215 * temperature
                - 11.37 * (wind_speed_kmh ** 0.16)
                + 0.3965 * temperature * (wind_speed_kmh ** 0.16)
            )
            return wind_chill
        else:
            return temperature
    
    def calculate_heat_index(
        self,
        temperature: float,
        humidity: float
    ) -> float:
        """
        Calculate heat index (apparent temperature)
        
        Args:
            temperature: Temperature in Celsius
            humidity: Relative humidity (0-100)
            
        Returns:
            Heat index
        """
        # Convert to Fahrenheit for calculation
        temp_f = temperature * 9/5 + 32
        
        # Heat index formula (valid for temp >= 80°F)
        if temp_f >= 80:
            hi = (
                -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
                - 0.22475541 * temp_f * humidity
                - 0.00683783 * temp_f**2
                - 0.05481717 * humidity**2
                + 0.00122874 * temp_f**2 * humidity
                + 0.00085282 * temp_f * humidity**2
                - 0.00000199 * temp_f**2 * humidity**2
            )
            # Convert back to Celsius
            return (hi - 32) * 5/9
        else:
            return temperature
    
    def extract_temporal_features(
        self,
        timestamp: datetime
    ) -> Dict[str, float]:
        """
        Extract temporal features from timestamp
        
        Args:
            timestamp: Datetime object
            
        Returns:
            Dictionary of temporal features
        """
        features = {
            'hour_of_day': timestamp.hour / 24.0,
            'day_of_week': timestamp.weekday() / 7.0,
            'day_of_month': timestamp.day / 31.0,
            'month_of_year': timestamp.month / 12.0,
            'is_weekend': float(timestamp.weekday() >= 5),
            # Cyclical encoding for hour
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            # Cyclical encoding for month
            'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
        }
        
        return features
    
    def extract_weather_features(
        self,
        weather_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract features from weather data
        
        Args:
            weather_data: Raw weather data dictionary
            
        Returns:
            Engineered weather features
        """
        features = {}
        
        # Basic features
        temperature = weather_data.get('temperature', 20.0)
        pressure = weather_data.get('pressure', 1013.0)
        humidity = weather_data.get('humidity', 50.0)
        wind_speed = weather_data.get('wind_speed', 0.0)
        
        # Normalized features
        features['temperature_norm'] = self.normalize_temperature(temperature)
        features['pressure_norm'] = self.normalize_pressure(pressure)
        features['humidity_norm'] = self.normalize_humidity(humidity)
        features['wind_speed_norm'] = min(wind_speed / 20.0, 1.0)  # Normalize to max 20 m/s
        
        # Raw features
        features['temperature'] = temperature
        features['pressure'] = pressure
        features['humidity'] = humidity
        features['wind_speed'] = wind_speed
        
        # Derived features
        features['wind_chill'] = self.calculate_wind_chill(temperature, wind_speed)
        features['heat_index'] = self.calculate_heat_index(temperature, humidity)
        
        # Discomfort indicators
        features['cold_stress'] = max(0, 10 - temperature) / 20.0  # Increases below 10°C
        features['heat_stress'] = max(0, temperature - 35) / 15.0  # Increases above 35°C
        features['pressure_anomaly'] = abs(pressure - 1013) / 50.0  # Deviation from standard
        
        return features
    
    def create_feature_vector(
        self,
        weather_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Create feature vector from weather and temporal data
        
        Args:
            weather_data: Current weather data
            timestamp: Timestamp for temporal features
            historical_data: Historical weather data for gradients
            
        Returns:
            Feature vector as numpy array
        """
        features = {}
        
        # Weather features
        features.update(self.extract_weather_features(weather_data))
        
        # Temporal features
        if timestamp:
            features.update(self.extract_temporal_features(timestamp))
        
        # Gradient features from historical data
        if historical_data and len(historical_data) > 1:
            pressure_history = [d.get('pressure', 1013.0) for d in historical_data]
            temp_history = [d.get('temperature', 20.0) for d in historical_data]
            time_intervals = list(range(len(historical_data)))  # Assuming uniform intervals
            
            features['pressure_gradient'] = self.calculate_pressure_gradient(
                pressure_history, time_intervals
            )
            features['temperature_gradient'] = self.calculate_temperature_gradient(
                temp_history, time_intervals
            )
        else:
            features['pressure_gradient'] = 0.0
            features['temperature_gradient'] = 0.0
        
        # Convert to array
        feature_vector = np.array(list(features.values()), dtype=np.float32)
        
        logger.debug(f"Created environmental feature vector: shape={feature_vector.shape}")
        return feature_vector
    
    def process_weather_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Process weather dataframe and add engineered features
        
        Args:
            df: Weather dataframe
            timestamp_column: Name of timestamp column
            
        Returns:
            Dataframe with additional engineered features
        """
        logger.info(f"Processing weather dataframe with {len(df)} rows")
        
        # Convert timestamp column to datetime if needed
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Extract weather features for each row
        weather_features = []
        for _, row in df.iterrows():
            weather_data = row.to_dict()
            features = self.extract_weather_features(weather_data)
            weather_features.append(features)
        
        # Add features to dataframe
        feature_df = pd.DataFrame(weather_features)
        result_df = pd.concat([df, feature_df], axis=1)
        
        # Calculate gradients (rolling window)
        if 'pressure' in result_df.columns:
            result_df['pressure_gradient'] = result_df['pressure'].diff()
        if 'temperature' in result_df.columns:
            result_df['temperature_gradient'] = result_df['temperature'].diff()
        
        logger.info(f"Added {len(feature_df.columns)} engineered features")
        return result_df
