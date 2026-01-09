"""
Xeno-Canto API Client for bird audio recordings
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from loguru import logger

from .base_client import BaseAPIClient


class XenoCantoClient(BaseAPIClient):
    """Client for Xeno-Canto API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit: int = 30,
        timeout: int = 60
    ):
        """
        Initialize Xeno-Canto client
        
        Args:
            api_key: API key (not required for Xeno-Canto but included for consistency)
            rate_limit: Maximum requests per minute
            timeout: Request timeout in seconds
        """
        # Xeno-Canto doesn't require API key but we keep it for consistency
        super().__init__(
            base_url="https://xeno-canto.org/api/2",
            api_key=api_key,
            rate_limit=rate_limit,
            timeout=timeout
        )
        
        logger.info("Xeno-Canto API client initialized")
    
    def search_recordings(
        self,
        query: str,
        page: int = 1,
        quality: str = "A"
    ) -> Dict[str, Any]:
        """
        Search for bird recordings
        
        Args:
            query: Search query (e.g., 'Corvus splendens' or 'gen:Corvus cnt:India')
            page: Page number (1-indexed)
            quality: Recording quality filter (A: excellent, B: good, C: fair, D-E: poor)
            
        Returns:
            Search results dictionary
        """
        try:
            endpoint = "recordings"
            
            # Build query with quality filter
            full_query = f"{query} q:{quality}" if quality else query
            
            params = {
                "query": full_query,
                "page": page
            }
            
            logger.info(f"Searching Xeno-Canto for: {full_query} (page {page})")
            data = self.get(endpoint, params=params)
            
            num_recordings = data.get('numRecordings', 0)
            logger.info(f"Found {num_recordings} recordings")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to search recordings: {str(e)}")
            return {"numRecordings": "0", "numPages": "0", "recordings": []}
    
    def get_recordings_by_species(
        self,
        scientific_name: str,
        country: str = "India",
        quality: str = "A",
        max_recordings: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recordings for a specific species
        
        Args:
            scientific_name: Scientific name of the species
            country: Country filter
            quality: Recording quality filter
            max_recordings: Maximum number of recordings to retrieve
            
        Returns:
            List of recording dictionaries
        """
        try:
            # Build query: species AND country
            query = f'"{scientific_name}" cnt:"{country}"'
            
            all_recordings = []
            page = 1
            
            while len(all_recordings) < max_recordings:
                result = self.search_recordings(query, page=page, quality=quality)
                
                recordings = result.get('recordings', [])
                if not recordings:
                    break
                
                all_recordings.extend(recordings)
                
                # Check if there are more pages
                num_pages = int(result.get('numPages', 0))
                if page >= num_pages:
                    break
                
                page += 1
            
            # Limit to max_recordings
            all_recordings = all_recordings[:max_recordings]
            
            logger.info(f"Retrieved {len(all_recordings)} recordings for {scientific_name}")
            return all_recordings
            
        except Exception as e:
            logger.error(f"Failed to get recordings for species: {str(e)}")
            return []
    
    def get_recording_metadata(self, recording_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for a specific recording
        
        Args:
            recording_id: Xeno-Canto recording ID
            
        Returns:
            Recording metadata dictionary
        """
        try:
            # Search by ID
            result = self.search_recordings(f"nr:{recording_id}")
            
            recordings = result.get('recordings', [])
            if recordings:
                logger.info(f"Retrieved metadata for recording {recording_id}")
                return recordings[0]
            else:
                logger.warning(f"No metadata found for recording {recording_id}")
                return {}
            
        except Exception as e:
            logger.error(f"Failed to get recording metadata: {str(e)}")
            return {}
    
    def download_recording(
        self,
        recording: Dict[str, Any],
        output_dir: Path,
        quality: str = "med"
    ) -> Optional[Path]:
        """
        Download a recording audio file
        
        Args:
            recording: Recording dictionary from API
            output_dir: Output directory for downloaded file
            quality: Audio quality ('low', 'med', 'high')
            
        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            # Get download URL based on quality
            if quality == "high" and "file" in recording:
                url = recording["file"]
            elif quality == "med" and "file-name" in recording:
                # Construct medium quality URL
                base_url = "https://xeno-canto.org/sounds/uploaded"
                file_name = recording["file-name"]
                # Medium quality typically has .mp3 extension
                url = f"{base_url}/{file_name}"
            else:
                url = recording.get("file", "")
            
            if not url:
                logger.warning(f"No download URL found for recording {recording.get('id', 'unknown')}")
                return None
            
            # Create filename from recording ID and species
            recording_id = recording.get('id', 'unknown')
            species = recording.get('gen', 'Unknown') + '_' + recording.get('sp', 'species')
            filename = f"XC{recording_id}_{species.replace(' ', '_')}.mp3"
            
            output_path = output_dir / filename
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Download file
            success = self.download_file(url, str(output_path), timeout=300)
            
            if success:
                logger.info(f"Downloaded recording to {output_path}")
                return output_path
            else:
                return None
            
        except Exception as e:
            logger.error(f"Failed to download recording: {str(e)}")
            return None
    
    def download_multiple_recordings(
        self,
        recordings: List[Dict[str, Any]],
        output_dir: Path,
        max_downloads: int = 10
    ) -> List[Path]:
        """
        Download multiple recordings
        
        Args:
            recordings: List of recording dictionaries
            output_dir: Output directory
            max_downloads: Maximum number of recordings to download
            
        Returns:
            List of paths to downloaded files
        """
        downloaded_paths = []
        
        for i, recording in enumerate(recordings[:max_downloads]):
            logger.info(f"Downloading recording {i+1}/{min(len(recordings), max_downloads)}")
            
            path = self.download_recording(recording, output_dir)
            if path:
                downloaded_paths.append(path)
        
        logger.info(f"Successfully downloaded {len(downloaded_paths)} recordings")
        return downloaded_paths
    
    def save_recordings_metadata(
        self,
        recordings: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """
        Save recording metadata to CSV file
        
        Args:
            recordings: List of recording dictionaries
            output_path: Output file path
        """
        try:
            if not recordings:
                logger.warning("No recordings metadata to save")
                return
            
            # Extract relevant fields
            metadata = []
            for rec in recordings:
                metadata.append({
                    'id': rec.get('id'),
                    'scientific_name': f"{rec.get('gen')} {rec.get('sp')}",
                    'common_name': rec.get('en'),
                    'country': rec.get('cnt'),
                    'location': rec.get('loc'),
                    'latitude': rec.get('lat'),
                    'longitude': rec.get('lng'),
                    'date': rec.get('date'),
                    'time': rec.get('time'),
                    'quality': rec.get('q'),
                    'length': rec.get('length'),
                    'recordist': rec.get('rec'),
                    'file_url': rec.get('file'),
                    'license': rec.get('lic'),
                    'remarks': rec.get('rmk', '')
                })
            
            df = pd.DataFrame(metadata)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved metadata for {len(recordings)} recordings to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save recordings metadata: {str(e)}")
    
    def get_recordings_by_location(
        self,
        latitude: float,
        longitude: float,
        species: Optional[str] = None,
        radius_km: int = 50,
        quality: str = "A"
    ) -> List[Dict[str, Any]]:
        """
        Get recordings near a geographic location
        
        Args:
            latitude: Latitude
            longitude: Longitude
            species: Optional species filter
            radius_km: Search radius in kilometers
            quality: Recording quality filter
            
        Returns:
            List of recording dictionaries
        """
        try:
            # Build query with location box
            # Approximate: 1 degree = 111 km
            lat_delta = radius_km / 111.0
            lon_delta = radius_km / (111.0 * abs(latitude) / 90.0) if latitude != 0 else radius_km / 111.0
            
            box_query = f"box:{latitude-lat_delta},{longitude-lon_delta},{latitude+lat_delta},{longitude+lon_delta}"
            
            if species:
                query = f'"{species}" {box_query}'
            else:
                query = box_query
            
            result = self.search_recordings(query, quality=quality)
            recordings = result.get('recordings', [])
            
            logger.info(f"Found {len(recordings)} recordings near ({latitude}, {longitude})")
            return recordings
            
        except Exception as e:
            logger.error(f"Failed to get recordings by location: {str(e)}")
            return []


# Convenience function
def create_xenocanto_client() -> XenoCantoClient:
    """Create and return a Xeno-Canto client instance"""
    return XenoCantoClient()
