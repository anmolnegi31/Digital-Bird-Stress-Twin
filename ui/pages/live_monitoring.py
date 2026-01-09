"""
Live Monitoring page
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show():
    st.title("🌍 Live Monitoring")
    st.markdown("Real-time bird stress monitoring dashboard.")
    
    # Auto-refresh
    st.markdown("🔄 Auto-refresh every 30 seconds")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Locations", "5", delta="2")
    with col2:
        st.metric("Avg Stress Level", "45%", delta="-5%")
    with col3:
        st.metric("High Risk Areas", "1", delta="0")
    with col4:
        st.metric("Total Observations", "1,234", delta="+156")
    
    st.markdown("---")
    
    # Simulated real-time chart
    st.subheader("📊 Real-Time Stress Trends")
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
    stress_values = np.random.uniform(30, 70, 24) + np.sin(np.linspace(0, 4*np.pi, 24)) * 10
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=stress_values,
        mode='lines+markers',
        name='Stress Level',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.update_layout(
        title="24-Hour Stress Level Trend",
        xaxis_title="Time",
        yaxis_title="Stress Level (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Alerts
    st.subheader("🚨 Recent Alerts")
    
    alerts = [
        {"time": "2 min ago", "location": "Delhi", "level": "HIGH", "message": "Stress level exceeded 70%"},
        {"time": "15 min ago", "location": "Mumbai", "level": "MEDIUM", "message": "Unusual vocalization pattern"},
        {"time": "1 hour ago", "location": "Bangalore", "level": "LOW", "message": "Normal stress levels resumed"}
    ]
    
    for alert in alerts:
        if alert["level"] == "HIGH":
            st.error(f"🔴 **{alert['location']}** ({alert['time']}): {alert['message']}")
        elif alert["level"] == "MEDIUM":
            st.warning(f"🟡 **{alert['location']}** ({alert['time']}): {alert['message']}")
        else:
            st.success(f"🟢 **{alert['location']}** ({alert['time']}): {alert['message']}")
