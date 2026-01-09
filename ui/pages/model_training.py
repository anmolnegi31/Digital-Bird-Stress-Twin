"""
Model Training page
"""

import streamlit as st
from pathlib import Path

def show():
    st.title("🤖 Model Training")
    st.markdown("Train LSTM and VAE models on collected data.")
    
    st.info("🚧 Model training interface coming soon! Use `python src/train.py` from command line for now.")
    
    st.markdown("""
    ### Training Steps:
    1. Prepare training data in `data/processed/`
    2. Configure hyperparameters below
    3. Start training with MLflow tracking
    4. Monitor progress in MLflow Dashboard
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("LSTM Parameters")
        st.number_input("Hidden Size", value=256)
        st.number_input("Num Layers", value=3)
        st.slider("Dropout", 0.0, 0.5, 0.3)
    
    with col2:
        st.subheader("Training Parameters")
        st.number_input("Batch Size", value=32)
        st.number_input("Epochs", value=100)
        st.number_input("Learning Rate", value=0.001, format="%.4f")
    
    if st.button("Start Training", type="primary"):
        st.warning("Please use CLI: `python src/train.py --data your_data.csv`")
