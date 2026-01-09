"""
Data Collection page for Digital Bird Stress Twin
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data_ingestion import create_ebird_client, create_xenocanto_client, create_weather_client

def show():
    """Display data collection page"""
    
    st.title("📊 Data Collection")
    st.markdown("Collect bird observations, audio recordings, and weather data from external APIs.")
    
    st.markdown("---")
    
    # API Status Check
    st.subheader("🔌 API Connectivity Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            client = create_ebird_client()
            st.success("✅ eBird API Connected")
        except Exception as e:
            st.error(f"❌ eBird API Error: {str(e)[:50]}")
    
    with col2:
        try:
            client = create_xenocanto_client()
            st.success("✅ Xeno-Canto API Connected")
        except Exception as e:
            st.error(f"❌ Xeno-Canto API Error: {str(e)[:50]}")
    
    with col3:
        try:
            client = create_weather_client()
            st.success("✅ OpenWeather API Connected")
        except Exception as e:
            st.error(f"❌ OpenWeather API Error: {str(e)[:50]}")
    
    st.markdown("---")
    
    # Data collection configuration
    st.subheader("⚙️ Collection Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🐦 Bird Observations")
        region_code = st.text_input("Region Code", value="IN", help="eBird region code (e.g., IN for India, US for United States)")
        species_code = st.text_input("Species Code (Optional)", value="", help="Leave empty for all species")
        days_back = st.slider("Days of Data", min_value=1, max_value=30, value=7)
        max_observations = st.number_input("Max Observations", min_value=10, max_value=1000, value=100, step=10)
    
    with col2:
        st.markdown("#### 🎵 Audio Recordings")
        species_name = st.text_input("Species Scientific Name", value="Corvus splendens", help="e.g., Corvus splendens (House Crow)")
        country_name = st.text_input("Country", value="India")
        audio_quality = st.selectbox("Audio Quality", ["A", "B", "C", "Any"], index=0, help="A = Best quality")
        max_recordings = st.number_input("Max Recordings", min_value=1, max_value=100, value=10, step=1)
    
    st.markdown("---")
    
    # Data type selection
    st.subheader("📋 Select Data Types to Collect")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        collect_birds = st.checkbox("🐦 Bird Observations", value=True)
    with col2:
        collect_audio = st.checkbox("🎵 Audio Recordings", value=True)
    with col3:
        collect_weather = st.checkbox("🌤️ Weather Data", value=True)
    
    st.markdown("---")
    
    # Collection button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Data Collection", use_container_width=True, type="primary"):
            
            output_dir = Path("data/raw")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {
                "birds": 0,
                "audio": 0,
                "weather": 0
            }
            
            total_steps = sum([collect_birds, collect_audio, collect_weather])
            current_step = 0
            
            # Collect bird observations
            if collect_birds:
                try:
                    status_text.text("🐦 Collecting bird observations...")
                    client = create_ebird_client()
                    
                    if species_code:
                        observations = client.get_recent_observations(
                            region_code=region_code,
                            species_code=species_code,
                            days=days_back,
                            max_results=max_observations
                        )
                    else:
                        observations = client.get_recent_observations(
                            region_code=region_code,
                            days=days_back,
                            max_results=max_observations
                        )
                    
                    if observations:
                        output_file = output_dir / f"ebird_observations_{region_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        client.save_observations_to_csv(observations, output_file)
                        results["birds"] = len(observations)
                        st.success(f"✅ Collected {len(observations)} bird observations")
                    else:
                        st.warning("⚠️ No bird observations found")
                    
                except Exception as e:
                    st.error(f"❌ Error collecting bird data: {str(e)}")
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Collect audio recordings
            if collect_audio:
                try:
                    status_text.text("🎵 Collecting audio recordings...")
                    client = create_xenocanto_client()
                    
                    quality = None if audio_quality == "Any" else audio_quality
                    
                    recordings = client.get_recordings_by_species(
                        scientific_name=species_name,
                        country=country_name,
                        quality=quality,
                        max_recordings=max_recordings
                    )
                    
                    if recordings:
                        # Save metadata
                        metadata_file = output_dir / f"xenocanto_metadata_{species_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        client.save_recordings_metadata(recordings, metadata_file)
                        
                        # Download audio files
                        audio_dir = output_dir / "audio"
                        audio_dir.mkdir(exist_ok=True)
                        
                        downloaded = client.download_multiple_recordings(
                            recordings=recordings[:min(5, len(recordings))],  # Limit to 5 for demo
                            output_dir=audio_dir
                        )
                        
                        results["audio"] = len(downloaded)
                        st.success(f"✅ Collected {len(downloaded)} audio recordings")
                    else:
                        st.warning("⚠️ No audio recordings found")
                    
                except Exception as e:
                    st.error(f"❌ Error collecting audio data: {str(e)}")
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Collect weather data
            if collect_weather:
                try:
                    status_text.text("🌤️ Collecting weather data...")
                    client = create_weather_client()
                    
                    # Predefined locations
                    locations = [
                        {'name': 'Delhi'},
                        {'name': 'Mumbai'},
                        {'name': 'Bangalore'},
                        {'name': 'Chennai'},
                        {'name': 'Kolkata'}
                    ]
                    
                    weather_data = []
                    
                    for location in locations:
                        try:
                            weather = client.get_current_weather(city=location['name'])
                            features = client.extract_weather_features(weather)
                            features['location'] = location['name']
                            features['timestamp'] = datetime.now().isoformat()
                            weather_data.append(features)
                        except:
                            continue
                    
                    if weather_data:
                        df = pd.DataFrame(weather_data)
                        output_file = output_dir / f"weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        df.to_csv(output_file, index=False)
                        results["weather"] = len(weather_data)
                        st.success(f"✅ Collected weather data for {len(weather_data)} locations")
                    else:
                        st.warning("⚠️ No weather data collected")
                    
                except Exception as e:
                    st.error(f"❌ Error collecting weather data: {str(e)}")
                
                current_step += 1
                progress_bar.progress(1.0)
            
            status_text.text("✅ Data collection completed!")
            
            # Show summary
            st.markdown("---")
            st.subheader("📊 Collection Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🐦 Bird Observations", results["birds"])
            with col2:
                st.metric("🎵 Audio Files", results["audio"])
            with col3:
                st.metric("🌤️ Weather Records", results["weather"])
    
    st.markdown("---")
    
    # Show collected files
    st.subheader("📁 Collected Data Files")
    
    data_path = Path("data/raw")
    
    if data_path.exists():
        # Bird observation files
        st.markdown("#### 🐦 Bird Observations")
        ebird_files = sorted(data_path.glob("ebird_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if ebird_files:
            for file in ebird_files[:5]:  # Show last 5
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                size_kb = file.stat().st_size / 1024
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.text(f"{size_kb:.1f} KB")
                with col3:
                    st.text(modified.strftime("%Y-%m-%d %H:%M"))
        else:
            st.info("No bird observation files yet")
        
        # Weather files
        st.markdown("#### 🌤️ Weather Data")
        weather_files = sorted(data_path.glob("weather_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if weather_files:
            for file in weather_files[:5]:  # Show last 5
                modified = datetime.fromtimestamp(file.stat().st_mtime)
                size_kb = file.stat().st_size / 1024
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.text(f"{size_kb:.1f} KB")
                with col3:
                    st.text(modified.strftime("%Y-%m-%d %H:%M"))
        else:
            st.info("No weather data files yet")
        
        # Audio files
        audio_path = data_path / "audio"
        if audio_path.exists():
            st.markdown("#### 🎵 Audio Recordings")
            audio_files = sorted(audio_path.glob("*.mp3"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if audio_files:
                st.text(f"Total audio files: {len(audio_files)}")
                for file in audio_files[:3]:  # Show last 3
                    modified = datetime.fromtimestamp(file.stat().st_mtime)
                    size_kb = file.stat().st_size / 1024
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.text(f"🎵 {file.name}")
                    with col2:
                        st.text(f"{size_kb:.1f} KB")
                    with col3:
                        st.text(modified.strftime("%Y-%m-%d %H:%M"))
            else:
                st.info("No audio files yet")
    else:
        st.info("No data directory found. Start collection to create it!")
