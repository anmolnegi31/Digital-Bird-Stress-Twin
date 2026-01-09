"""
Predictions page
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

def show():
    st.title("📈 Stress Predictions")
    st.markdown("Make real-time predictions using trained models.")
    
    st.info("💡 Load a trained model to start making predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        location = st.text_input("Location", value="Delhi")
        species = st.selectbox("Species", ["House Crow", "House Sparrow", "Common Myna"])
        temperature = st.slider("Temperature (°C)", -10, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        wind_speed = st.slider("Wind Speed (m/s)", 0, 30, 5)
    
    with col2:
        st.subheader("Model Selection")
        model_type = st.radio("Model", ["LSTM", "VAE"])
        confidence_level = st.slider("Confidence Level", 0.5, 0.99, 0.95)
        
        if st.button("🔮 Predict Stress Level", type="primary", use_container_width=True):
            with st.spinner("Generating prediction..."):
                # Simulated prediction
                stress_level = np.random.uniform(0.3, 0.8)
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=stress_level * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Stress Level (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                if stress_level < 0.3:
                    st.success("✅ LOW STRESS - Birds are in good condition")
                elif stress_level < 0.6:
                    st.warning("⚠️ MEDIUM STRESS - Monitor closely")
                else:
                    st.error("🚨 HIGH STRESS - Immediate attention needed")
