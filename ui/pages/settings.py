"""
Settings page
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def show():
    st.title("⚙️ Settings")
    st.markdown("Configure API keys and system parameters.")
    
    env_path = Path(".env")
    load_dotenv(dotenv_path=env_path)
    
    # API Configuration
    st.subheader("🔑 API Keys")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ebird_key = st.text_input(
            "eBird API Key",
            value=os.getenv("EBIRD_API_KEY", ""),
            type="password"
        )
        
        xeno_key = st.text_input(
            "Xeno-Canto API Key",
            value=os.getenv("XENO_CANTO_API_KEY", ""),
            type="password"
        )
    
    with col2:
        weather_key = st.text_input(
            "OpenWeatherMap API Key",
            value=os.getenv("OPENWEATHER_API_KEY", ""),
            type="password"
        )
    
    if st.button("💾 Save API Keys", type="primary"):
        try:
            if not env_path.exists():
                env_path.touch()
            
            set_key(env_path, "EBIRD_API_KEY", ebird_key)
            set_key(env_path, "XENO_CANTO_API_KEY", xeno_key)
            set_key(env_path, "OPENWEATHER_API_KEY", weather_key)
            
            st.success("✅ API keys saved successfully!")
        except Exception as e:
            st.error(f"❌ Error saving keys: {str(e)}")
    
    st.markdown("---")
    
    # Model Configuration
    st.subheader("🤖 Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input("LSTM Hidden Size", value=256, step=32)
        st.number_input("Number of Layers", value=3, min_value=1, max_value=10)
        st.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05)
    
    with col2:
        st.number_input("Batch Size", value=32, step=8)
        st.number_input("Learning Rate", value=0.001, format="%.4f")
        st.number_input("Max Epochs", value=100, step=10)
    
    st.markdown("---")
    
    # System Info
    st.subheader("💻 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Data Path:**\n{Path('data').absolute()}")
    
    with col2:
        st.info(f"**Model Path:**\n{Path('models').absolute()}")
    
    with col3:
        st.info(f"**Logs Path:**\n{Path('logs').absolute()}")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Digital Bird Stress Twin v1.0.0**
    
    An AI-powered system for monitoring avian stress levels using:
    - Deep Learning (LSTM + VAE)
    - Real-world APIs (eBird, Xeno-Canto, OpenWeather)
    - MLOps best practices (MLflow, DVC, Evidently)
    
    Built with ❤️ using Python, PyTorch, FastAPI, and Streamlit.
    """)
