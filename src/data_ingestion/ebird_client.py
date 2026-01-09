"""
eBird API Client for bird observation data
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

from .base_client import BaseAPIClient


class EBirdClient(BaseAPIClient):
    """Client for eBird API v2"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 10,
        timeout: int = 30
    ):
        """
        Initialize eBird client
        
        Args:
            api_key: eBird API key
            rate_limit: Maximum requests per minute
            timeout: Request timeout in seconds
        """
        if api_key is None:
            api_key = os.getenv("EBIRD_API_KEY")
        
        if not api_key:
            raise ValueError("eBird API key is required")
        
        super().__init__(
            base_url="https://api.ebird.org/v2",
            api_key=api_key,
            rate_limit=rate_limit,
            timeout=timeout
        )
        
        logger.info("eBird API client initialized")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with eBird-specific format"""
        headers = super()._get_headers()
        # eBird uses X-eBirdApiToken instead of Bearer
        if self.api_key:
            headers.pop("Authorization", None)
            headers["X-eBirdApiToken"] = self.api_key
        return headers
    
    def get_recent_observations(
        self,
        region_code: str = "IN",
        species_code: Optional[str] = None,
        days: int = 14,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent bird observations for a region
        
        Args:
            region_code: Region code (e.g., 'IN' for India)
            species_code: Optional species code filter
            days: Number of days back to search
            max_results: Maximum number of results
            
        Returns:
            List of observation dictionaries
        """
        try:
            if species_code:
                endpoint = f"data/obs/{region_code}/recent/{species_code}"
            else:
                endpoint = f"data/obs/{region_code}/recent"
            
            params = {
                "back": days,
                "maxResults": max_results
            }
            
            logger.info(f"Fetching recent observations for region {region_code}")
            data = self.get(endpoint, params=params)
            
            logger.info(f"Retrieved {len(data)} observations")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get recent observations: {str(e)}")
            return []
    
    def get_notable_observations(
        self,
        region_code: str = "IN",
        days: int = 14,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get notable (rare/unusual) bird observations
        
        Args:
            region_code: Region code
            days: Number of days back to search
            max_results: Maximum number of results
            
        Returns:
            List of notable observation dictionaries
        """
        try:
            endpoint = f"data/obs/{region_code}/recent/notable"
            
            params = {
                "back": days,
                "maxResults": max_results
            }
            
            logger.info(f"Fetching notable observations for region {region_code}")
            data = self.get(endpoint, params=params)
            
            logger.info(f"Retrieved {len(data)} notable observations")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get notable observations: {str(e)}")
            return []
    
    def get_nearby_observations(
        self,
        lat: float,
        lon: float,
        radius_km: int = 25,
        days: int = 14,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get observations near a location
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            days: Number of days back to search
            max_results: Maximum number of results
            
        Returns:
            List of observation dictionaries
        """
        try:
            endpoint = "data/obs/geo/recent"
            
            params = {
                "lat": lat,
                "lng": lon,
                "dist": radius_km,
                "back": days,
                "maxResults": max_results
            }
            
            logger.info(f"Fetching observations near ({lat}, {lon})")
            data = self.get(endpoint, params=params)
            
            logger.info(f"Retrieved {len(data)} nearby observations")
            return data
            
        except Exception as e:
            logger.error(f"Failed to get nearby observations: {str(e)}")
            return []
    
    def get_species_info(self, species_code: str, region_code: str = "IN") -> Dict[str, Any]:
        """
        Get information about a specific species in a region
        
        Args:
            species_code: Species code
            region_code: Region code
            
        Returns:
            Species information dictionary
        """
        try:
            endpoint = f"ref/taxonomy/ebird"
            params = {"species": species_code, "fmt": "json"}
            
            logger.info(f"Fetching species info for {species_code}")
            data = self.get(endpoint, params=params)
            
            if data:
                logger.info(f"Retrieved species info for {data[0].get('comName', 'Unknown')}")
                return data[0]
            else:
                logger.warning(f"No species info found for {species_code}")
                return {}
            
        except Exception as e:
            logger.error(f"Failed to get species info: {str(e)}")
            return {}
    
    def save_observations_to_csv(
        self,
        observations: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        Save observations to CSV file
        
        Args:
            observations: List of observation dictionaries
            output_path: Output file path
        """
        try:
            if not observations:
                logger.warning("No observations to save")
                return
            
            df = pd.DataFrame(observations)
            df['fetchedAt'] = datetime.now().isoformat()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(observations)} observations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save observations: {str(e)}")
    
    def get_observations_for_species_list(
        self,
        species_codes: List[str],
        region_code: str = "IN",
        days: int = 14
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get observations for multiple species
        
        Args:
            species_codes: List of species codes
            region_code: Region code
            days: Number of days back to search
            
        Returns:
            Dictionary mapping species codes to observations
        """
        all_observations = {}
        
        for species_code in species_codes:
            logger.info(f"Fetching observations for {species_code}")
            observations = self.get_recent_observations(
                region_code=region_code,
                species_code=species_code,
                days=days
            )
            all_observations[species_code] = observations
        
        return all_observations


# Convenience function
def create_ebird_client() -> EBirdClient:
    """Create and return an eBird client instance"""
    return EBirdClient()
