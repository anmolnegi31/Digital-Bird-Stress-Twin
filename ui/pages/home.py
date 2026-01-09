"""
Home page for Digital Bird Stress Twin
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

def show():
    """Display home page"""
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### Welcome to Digital Bird Stress Twin! 👋
        
        An advanced AI system that monitors and predicts stress levels in bird populations using:
        - 🎵 **Audio Analysis** - Acoustic feature extraction from bird calls
        - 🌤️ **Environmental Data** - Weather and habitat conditions
        - 🤖 **Deep Learning** - LSTM + VAE hybrid models
        - 📊 **Real-time Monitoring** - Continuous stress level tracking
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Check for collected data
    data_path = Path("data/raw")
    ebird_files = list(data_path.glob("ebird_*.csv"))
    weather_files = list(data_path.glob("weather_*.csv"))
    audio_files = list((data_path / "audio").glob("*.mp3")) if (data_path / "audio").exists() else []
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #667eea; margin: 0;'>🐦 Bird Observations</h3>
            <h2 style='margin: 0.5rem 0;'>{}</h2>
            <p style='color: #666; margin: 0;'>Data files collected</p>
        </div>
        """.format(len(ebird_files)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #43A047; margin: 0;'>🌤️ Weather Records</h3>
            <h2 style='margin: 0.5rem 0;'>{}</h2>
            <p style='color: #666; margin: 0;'>Data files collected</p>
        </div>
        """.format(len(weather_files)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #FF6F00; margin: 0;'>🎵 Audio Files</h3>
            <h2 style='margin: 0.5rem 0;'>{}</h2>
            <p style='color: #666; margin: 0;'>Recordings available</p>
        </div>
        """.format(len(audio_files)), unsafe_allow_html=True)
    
    with col4:
        # Check for models
        model_path = Path("models/checkpoints")
        model_files = list(model_path.glob("*.pth")) if model_path.exists() else []
        
        st.markdown("""
        <div class="metric-card">
            <h3 style='color: #E91E63; margin: 0;'>🤖 Trained Models</h3>
            <h2 style='margin: 0.5rem 0;'>{}</h2>
            <p style='color: #666; margin: 0;'>Models ready</p>
        </div>
        """.format(len(model_files)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Recent Activity")
        
        if ebird_files:
            latest_ebird = max(ebird_files, key=lambda x: x.stat().st_mtime)
            modified_time = datetime.fromtimestamp(latest_ebird.stat().st_mtime)
            st.success(f"✅ Latest data collection: {modified_time.strftime('%Y-%m-%d %H:%M')}")
            
            # Load and show preview
            try:
                df = pd.read_csv(latest_ebird)
                st.info(f"📊 {len(df)} bird observations in latest file")
                
                with st.expander("View Recent Observations"):
                    st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load data preview: {str(e)}")
        else:
            st.info("ℹ️ No data collected yet. Go to **Data Collection** page to start!")
    
    with col2:
        st.subheader("🎯 Quick Actions")
        
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px;'>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Collect New Data", use_container_width=True):
            st.switch_page("pages/data_collection.py")
        
        if st.button("🧪 Analyze Existing Data", use_container_width=True):
            st.switch_page("pages/data_analysis.py")
        
        if st.button("🤖 Train Model", use_container_width=True):
            st.switch_page("pages/model_training.py")
        
        if st.button("📈 Make Predictions", use_container_width=True):
            st.switch_page("pages/predictions.py")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System architecture
    st.subheader("🏗️ System Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📥 Data Layer
        - eBird API Integration
        - Xeno-Canto Audio
        - OpenWeatherMap API
        - Feature Engineering
        """)
    
    with col2:
        st.markdown("""
        #### 🧠 Model Layer
        - Bidirectional LSTM
        - Attention Mechanism
        - Conditional VAE
        - MLflow Tracking
        """)
    
    with col3:
        st.markdown("""
        #### 🎯 Application Layer
        - FastAPI Backend
        - Streamlit UI
        - Real-time Monitoring
        - Drift Detection
        """)
    
    # Getting started guide
    st.markdown("---")
    st.subheader("🚀 Getting Started")
    
    with st.expander("📚 Step-by-Step Guide", expanded=False):
        st.markdown("""
        ### 1️⃣ Collect Data
        Navigate to the **Data Collection** page and:
        - Configure your region (default: India - IN)
        - Set time range for observations
        - Select data types (birds, audio, weather)
        - Click "Start Collection"
        
        ### 2️⃣ Analyze Data
        Go to **Data Analysis** to:
        - Explore collected observations
        - Visualize bird species distribution
        - Analyze environmental conditions
        - Extract audio features
        
        ### 3️⃣ Train Models
        In the **Model Training** page:
        - Prepare training dataset
        - Configure model hyperparameters
        - Start training with MLflow tracking
        - Monitor training progress
        
        ### 4️⃣ Make Predictions
        Use the **Predictions** page to:
        - Load trained models
        - Input new observations
        - Get stress level predictions
        - View confidence intervals
        
        ### 5️⃣ Monitor Performance
        Check **Live Monitoring** for:
        - Real-time stress trends
        - Geographic distribution maps
        - Alert notifications
        - Model drift detection
        """)
    
    # Additional resources
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        📖 **Documentation**
        
        Check out the README.md for detailed technical documentation.
        """)
    
    with col2:
        st.success("""
        💡 **Need Help?**
        
        Visit the Settings page for API configuration and troubleshooting.
        """)
    
    with col3:
        st.warning("""
        🔬 **Research Mode**
        
        Access MLflow Dashboard for experiment tracking and model comparison.
        """)
