"""
Quick Test: Collect Historical Disaster Data
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path BEFORE imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
load_dotenv()

from data_ingestion import create_disaster_client
from loguru import logger

def test_disaster_collection():
    """Test disaster data collection"""
    logger.info("Testing disaster data collection...")
    
    # Create client
    client = create_disaster_client()
    
    # Get last 5 years of earthquakes in India
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    logger.info(f"Collecting earthquakes from {start_date.date()} to {end_date.date()}")
    
    disasters = client.get_india_earthquakes(
        start_date=start_date,
        end_date=end_date,
        min_magnitude=3.0
    )
    
    logger.info(f"Found {len(disasters)} earthquake records")
    
    if disasters:
        # Show sample
        logger.info("\nSample disasters:")
        for d in disasters[:5]:
            logger.info(f"  - {d['timestamp'].date()}: Magnitude {d['magnitude']:.1f} at {d['place']}")
        
        # Create dataset
        disaster_dataset = client.create_disaster_dataset(
            disasters=disasters,
            window_days=7
        )
        
        # Save
        output_dir = Path("data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"disasters_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        disaster_dataset.to_csv(output_file, index=False)
        
        logger.info(f"\n✅ Saved disaster dataset to: {output_file}")
        logger.info(f"   Total events: {len(disaster_dataset)}")
        logger.info(f"   Date range: {disaster_dataset['disaster_timestamp'].min()} to {disaster_dataset['disaster_timestamp'].max()}")
        
        return True
    else:
        logger.warning("No disasters found!")
        return False

if __name__ == "__main__":
    try:
        success = test_disaster_collection()
        if success:
            print("\n🎉 SUCCESS! Disaster data collection works!")
            print("\nNext steps:")
            print("1. Collect bird/weather data: python scripts/collect_data.py --days 30 --collect birds,weather")
            print("2. Extract features from audio")
            print("3. Create time series dataset with labels")
            print("4. Train LSTM model")
        else:
            print("\n⚠️ No disaster data found. Check your internet connection and try again.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
