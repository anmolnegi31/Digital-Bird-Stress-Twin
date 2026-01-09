"""
Data collection script for Digital Bird Stress Twin
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_ingestion import (
    create_ebird_client,
    create_xenocanto_client,
    create_weather_client,
    create_disaster_client
)
from utils.config import Config


def collect_bird_observations(
    region: str,
    days: int,
    output_dir: Path
) -> None:
    """
    Collect bird observations from eBird
    
    Args:
        region: Region code (e.g., 'IN' for India)
        days: Number of days of data to collect
        output_dir: Output directory
    """
    logger.info(f"Collecting bird observations for region {region}")
    
    # Create client
    client = create_ebird_client()
    
    # Get observations
    observations = client.get_recent_observations(
        region_code=region,
        days=days,
        max_results=500
    )
    
    # Save to CSV
    output_file = output_dir / f"ebird_observations_{region}_{datetime.now().strftime('%Y%m%d')}.csv"
    client.save_observations_to_csv(observations, output_file)
    
    logger.info(f"Saved {len(observations)} observations to {output_file}")


def collect_audio_recordings(
    species: str,
    country: str,
    max_recordings: int,
    output_dir: Path
) -> None:
    """
    Collect audio recordings from Xeno-Canto
    
    Args:
        species: Species scientific name
        country: Country name
        max_recordings: Maximum number of recordings
        output_dir: Output directory
    """
    logger.info(f"Collecting audio recordings for {species}")
    
    # Create client
    client = create_xenocanto_client()
    
    # Search recordings
    recordings = client.get_recordings_by_species(
        scientific_name=species,
        country=country,
        quality="A",
        max_recordings=max_recordings
    )
    
    # Save metadata
    metadata_file = output_dir / f"xenocanto_metadata_{species.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
    client.save_recordings_metadata(recordings, metadata_file)
    
    # Download audio files (limit to first 10 for demo)
    audio_dir = output_dir / "audio"
    downloaded = client.download_multiple_recordings(
        recordings=recordings,
        output_dir=audio_dir,
        max_downloads=min(10, len(recordings))
    )
    
    logger.info(f"Downloaded {len(downloaded)} audio files to {audio_dir}")


def collect_weather_data(
    locations: list,
    output_dir: Path
) -> None:
    """
    Collect weather data for multiple locations
    
    Args:
        locations: List of location dictionaries
        output_dir: Output directory
    """
    logger.info(f"Collecting weather data for {len(locations)} locations")
    
    # Create client
    client = create_weather_client()
    
    # Get weather data
    all_weather_data = []
    
    for location in locations:
        name = location.get('name', 'Unknown')
        
        try:
            # Current weather
            weather = client.get_current_weather(city=name)
            
            # Extract features
            features = client.extract_weather_features(weather)
            features['location'] = name
            features['timestamp'] = datetime.now().isoformat()
            
            all_weather_data.append(features)
            
            logger.info(f"Collected weather for {name}")
            
        except Exception as e:
            logger.error(f"Failed to collect weather for {name}: {str(e)}")
    
    # Save to CSV
    df = pd.DataFrame(all_weather_data)
    output_file = output_dir / f"weather_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved weather data to {output_file}")


def collect_disaster_data(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path
) -> None:
    """
    Collect historical disaster data (earthquakes) from USGS
    
    Args:
        start_date: Start date for collection
        end_date: End date for collection
        output_dir: Output directory
    """
    logger.info(f"Collecting disaster data from {start_date.date()} to {end_date.date()}")
    
    # Create client
    client = create_disaster_client()
    
    # Get India earthquakes
    disasters = client.get_india_earthquakes(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=3.0
    )
    
    if not disasters:
        logger.warning("No disasters found in the specified period")
        return
    
    # Create disaster dataset with time windows
    disaster_dataset = client.create_disaster_dataset(
        disasters=disasters,
        window_days=7
    )
    
    # Save disaster records
    output_file = output_dir / f"disasters_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    disaster_dataset.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(disasters)} disaster records to {output_file}")
    
    # Also save detailed records
    detailed_file = output_dir / f"disasters_detailed_{datetime.now().strftime('%Y%m%d')}.csv"
    client.save_disasters_to_csv(disasters, detailed_file)


def main():
    """Main data collection script"""
    parser = argparse.ArgumentParser(description="Collect data for Digital Bird Stress Twin")
    parser.add_argument('--region', type=str, default='IN', help='Region code for eBird')
    parser.add_argument('--days', type=int, default=14, help='Number of days of data')
    parser.add_argument('--species', type=str, default='Corvus splendens', help='Species scientific name')
    parser.add_argument('--country', type=str, default='India', help='Country name')
    parser.add_argument('--max-recordings', type=int, default=50, help='Max audio recordings')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--collect', nargs='+', default=['all'], 
                       choices=['all', 'birds', 'audio', 'weather', 'disasters'],
                       help='What data to collect')
    parser.add_argument('--disaster-years', type=int, default=5, help='Years of historical disaster data')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config for locations
    config = Config()
    locations = config.get('data_ingestion.locations', [
        {'name': 'Delhi'},
        {'name': 'Mumbai'},
        {'name': 'Bangalore'}
    ])
    
    # Collect data based on arguments
    collect_types = args.collect
    
    if 'all' in collect_types or 'birds' in collect_types:
        collect_bird_observations(args.region, args.days, output_dir)
    
    if 'all' in collect_types or 'audio' in collect_types:
        collect_audio_recordings(args.species, args.country, args.max_recordings, output_dir)
    
    if 'all' in collect_types or 'weather' in collect_types:
        collect_weather_data(locations, output_dir)
    
    if 'all' in collect_types or 'disasters' in collect_types:
        # Collect historical disasters
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.disaster_years * 365)
        collect_disaster_data(start_date, end_date, output_dir)
    
    logger.info("Data collection completed successfully!")


if __name__ == "__main__":
    main()
