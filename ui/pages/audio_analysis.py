"""
Audio Analysis page
"""

import streamlit as st
from pathlib import Path

def show():
    st.title("🎵 Audio Analysis")
    st.markdown("Analyze bird call recordings and extract acoustic features.")
    
    audio_path = Path("data/raw/audio")
    
    if audio_path.exists():
        audio_files = list(audio_path.glob("*.mp3"))
        
        if audio_files:
            st.success(f"✅ Found {len(audio_files)} audio files")
            
            selected_audio = st.selectbox("Select Audio File", audio_files, format_func=lambda x: x.name)
            
            if selected_audio:
                st.audio(str(selected_audio))
                
                st.markdown("---")
                
                if st.button("🔬 Analyze Audio", type="primary"):
                    with st.spinner("Extracting features..."):
                        st.info("🚧 Audio feature extraction coming soon!")
                        
                        st.markdown("""
                        ### Features to Extract:
                        - 🎼 40 MFCC coefficients
                        - 📊 Spectral centroid, bandwidth, rolloff
                        - 🔊 Zero-crossing rate
                        - 🎵 Chroma features
                        - 📈 RMS energy
                        """)
        else:
            st.info("No audio files found. Collect audio data first!")
    else:
        st.warning("Audio directory not found. Collect audio data to analyze.")
