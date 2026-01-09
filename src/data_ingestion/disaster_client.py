"""
Disaster Data Client for earthquake, flood, and storm records
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger
import requests

from .base_client import BaseAPIClient


class DisasterClient(BaseAPIClient):
    """Client for USGS Earthquake API and disaster databases"""
    
    def __init__(
        self,
        rate_limit: int = 10,
        timeout: int = 30
    ):
        """
        Initialize Disaster client
        
        Args:
            rate_limit: Maximum requests per minute
            timeout: Request timeout in seconds
        """
        super().__init__(
            base_url="https://earthquake.usgs.gov/fdsnws/event/1",
            api_key=None,  # USGS API is public
            rate_limit=rate_limit,
            timeout=timeout
        )
        
        logger.info("Disaster API client initialized")
    
    def get_earthquakes(
        self,
        start_date: datetime,
        end_date: datetime,
        min_latitude: float,
        max_latitude: float,
        min_longitude: float,
        max_longitude: float,
        min_magnitude: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Get earthquake records from USGS
        
        Args:
            start_date: Start date for search
            end_date: End date for search
            min_latitude: Minimum latitude
            max_latitude: Maximum latitude
            min_longitude: Minimum longitude
            max_longitude: Maximum longitude
            min_magnitude: Minimum earthquake magnitude
            
        Returns:
            List of earthquake dictionaries
        """
        try:
            endpoint = "query"
            
            params = {
                "format": "geojson",
                "starttime": start_date.strftime("%Y-%m-%d"),
                "endtime": end_date.strftime("%Y-%m-%d"),
                "minlatitude": min_latitude,
                "maxlatitude": max_latitude,
                "minlongitude": min_longitude,
                "maxlongitude": max_longitude,
                "minmagnitude": min_magnitude,
                "orderby": "time"
            }
            
            logger.info(f"Fetching earthquakes from {start_date.date()} to {end_date.date()}")
            data = self.get(endpoint, params=params)
            
            # Extract earthquake features
            earthquakes = []
            if data and "features" in data:
                for feature in data["features"]:
                    props = feature["properties"]
                    coords = feature["geometry"]["coordinates"]
                    
                    eq = {
                        "timestamp": datetime.fromtimestamp(props["time"] / 1000),
                        "magnitude": props.get("mag", 0),
                        "place": props.get("place", "Unknown"),
                        "latitude": coords[1],
                        "longitude": coords[0],
                        "depth_km": coords[2],
                        "type": props.get("type", "earthquake"),
                        "title": props.get("title", ""),
                        "disaster_type": "earthquake"
                    }
                    earthquakes.append(eq)
            
            logger.info(f"Retrieved {len(earthquakes)} earthquake records")
            return earthquakes
            
        except Exception as e:
            logger.error(f"Failed to get earthquakes: {str(e)}")
            return []
    
    def get_india_earthquakes(
        self,
        start_date: datetime,
        end_date: datetime,
        min_magnitude: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Get earthquakes in India region
        
        Args:
            start_date: Start date
            end_date: End date
            min_magnitude: Minimum magnitude
            
        Returns:
            List of earthquake records
        """
        # India bounding box
        return self.get_earthquakes(
            start_date=start_date,
            end_date=end_date,
            min_latitude=6.0,
            max_latitude=37.0,
            min_longitude=68.0,
            max_longitude=98.0,
            min_magnitude=min_magnitude
        )
    
    def get_disasters_for_location(
        self,
        latitude: float,
        longitude: float,
        start_date: datetime,
        end_date: datetime,
        radius_km: float = 100
    ) -> List[Dict[str, Any]]:
        """
        Get disasters near a specific location
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date
            end_date: End date
            radius_km: Search radius in km
            
        Returns:
            List of disaster records
        """
        try:
            # Calculate bounding box from radius
            # Rough approximation: 1 degree ≈ 111 km
            lat_delta = radius_km / 111.0
            lng_delta = radius_km / (111.0 * abs(latitude))
            
            earthquakes = self.get_earthquakes(
                start_date=start_date,
                end_date=end_date,
                min_latitude=latitude - lat_delta,
                max_latitude=latitude + lat_delta,
                min_longitude=longitude - lng_delta,
                max_longitude=longitude + lng_delta,
                min_magnitude=3.0
            )
            
            return earthquakes
            
        except Exception as e:
            logger.error(f"Failed to get disasters for location: {str(e)}")
            return []
    
    def calculate_disaster_labels(
        self,
        disaster_timestamp: datetime,
        data_timestamps: List[datetime]
    ) -> List[float]:
        """
        Calculate stress labels based on hours before disaster
        
        Args:
            disaster_timestamp: When disaster occurred
            data_timestamps: List of data timestamps
            
        Returns:
            List of stress labels (0.0-1.0)
        """
        labels = []
        
        for ts in data_timestamps:
            hours_before = (disaster_timestamp - ts).total_seconds() / 3600
            
            if hours_before < 0:  # After disaster
                stress = 1.0
            elif hours_before <= 24:  # 0-24h before
                # Linear increase from 0.7 to 1.0
                stress = 0.7 + (0.3 * (24 - hours_before) / 24)
            elif hours_before <= 48:  # 24-48h before
                # Linear increase from 0.5 to 0.7
                stress = 0.5 + (0.2 * (48 - hours_before) / 24)
            elif hours_before <= 72:  # 48-72h before
                # Linear increase from 0.3 to 0.5
                stress = 0.3 + (0.2 * (72 - hours_before) / 24)
            elif hours_before <= 168:  # 3-7 days before
                # Gradual increase from 0.1 to 0.3
                stress = 0.1 + (0.2 * (168 - hours_before) / 96)
            else:  # More than 7 days before
                stress = 0.1  # Baseline stress
            
            labels.append(stress)
        
        return labels
    
    def create_disaster_dataset(
        self,
        disasters: List[Dict[str, Any]],
        window_days: int = 7
    ) -> pd.DataFrame:
        """
        Create labeled dataset from disaster records
        
        Args:
            disasters: List of disaster dictionaries
            window_days: Days before disaster to include
            
        Returns:
            DataFrame with disaster information and time windows
        """
        dataset = []
        
        for disaster in disasters:
            disaster_time = disaster["timestamp"]
            start_time = disaster_time - timedelta(days=window_days)
            
            record = {
                "disaster_id": f"{disaster['type']}_{disaster_time.strftime('%Y%m%d%H%M')}",
                "disaster_type": disaster["disaster_type"],
                "disaster_timestamp": disaster_time,
                "magnitude": disaster.get("magnitude", 0),
                "latitude": disaster["latitude"],
                "longitude": disaster["longitude"],
                "place": disaster["place"],
                "window_start": start_time,
                "window_end": disaster_time,
                "window_hours": window_days * 24
            }
            dataset.append(record)
        
        df = pd.DataFrame(dataset)
        logger.info(f"Created disaster dataset with {len(df)} events")
        
        return df
    
    def save_disasters_to_csv(
        self,
        disasters: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        Save disaster records to CSV
        
        Args:
            disasters: List of disaster dictionaries
            output_path: Output file path
        """
        try:
            if not disasters:
                logger.warning("No disasters to save")
                return
            
            df = pd.DataFrame(disasters)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(disasters)} disaster records to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save disasters: {str(e)}")


def create_disaster_client() -> DisasterClient:
    """Create and return a Disaster client instance"""
    return DisasterClient()
