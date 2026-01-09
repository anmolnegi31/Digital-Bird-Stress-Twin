"""
Data Analysis page for Digital Bird Stress Twin
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import numpy as np

def show():
    """Display data analysis page"""
    
    st.title("🔬 Data Analysis")
    st.markdown("Explore and visualize collected bird observations and environmental data.")
    
    st.markdown("---")
    
    # File selection
    data_path = Path("data/raw")
    
    if not data_path.exists():
        st.warning("⚠️ No data directory found. Please collect data first!")
        return
    
    # Get available files
    ebird_files = sorted(data_path.glob("ebird_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    weather_files = sorted(data_path.glob("weather_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not ebird_files and not weather_files:
        st.info("ℹ️ No data files found. Go to Data Collection page to gather data.")
        return
    
    # Tab selection
    tab1, tab2, tab3 = st.tabs(["🐦 Bird Observations", "🌤️ Weather Data", "📊 Combined Analysis"])
    
    # Bird Observations Tab
    with tab1:
        if ebird_files:
            st.subheader("📊 Bird Observation Analysis")
            
            # File selector
            selected_file = st.selectbox(
                "Select Data File",
                options=ebird_files,
                format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
            )
            
            # Load data
            try:
                df = pd.read_csv(selected_file)
                
                # Data overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Observations", len(df))
                with col2:
                    st.metric("Unique Species", df['comName'].nunique() if 'comName' in df.columns else 0)
                with col3:
                    st.metric("Unique Locations", df['locName'].nunique() if 'locName' in df.columns else 0)
                with col4:
                    st.metric("Date Range", f"{df['obsDt'].nunique() if 'obsDt' in df.columns else 0} days")
                
                st.markdown("---")
                
                # Species distribution
                if 'comName' in df.columns:
                    st.subheader("🐦 Species Distribution")
                    
                    species_counts = df['comName'].value_counts().head(15)
                    
                    fig = px.bar(
                        x=species_counts.values,
                        y=species_counts.index,
                        orientation='h',
                        labels={'x': 'Number of Observations', 'y': 'Species'},
                        title="Top 15 Most Observed Species",
                        color=species_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Location analysis
                if 'locName' in df.columns and 'lat' in df.columns and 'lng' in df.columns:
                    st.subheader("📍 Geographic Distribution")
                    
                    # Count observations by location
                    location_data = df.groupby(['locName', 'lat', 'lng']).size().reset_index(name='count')
                    
                    fig = px.scatter_mapbox(
                        location_data,
                        lat='lat',
                        lon='lng',
                        size='count',
                        hover_name='locName',
                        hover_data={'count': True, 'lat': ':.4f', 'lng': ':.4f'},
                        title="Observation Locations",
                        zoom=4,
                        height=500
                    )
                    fig.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Temporal analysis
                if 'obsDt' in df.columns:
                    st.subheader("📅 Temporal Patterns")
                    
                    df['obsDt'] = pd.to_datetime(df['obsDt'])
                    daily_counts = df.groupby(df['obsDt'].dt.date).size().reset_index(name='count')
                    daily_counts.columns = ['date', 'count']
                    
                    fig = px.line(
                        daily_counts,
                        x='date',
                        y='count',
                        title="Daily Observation Counts",
                        labels={'date': 'Date', 'count': 'Number of Observations'}
                    )
                    fig.update_traces(line_color='#667eea', line_width=3)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Raw data table
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"processed_{selected_file.name}",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
        else:
            st.info("No bird observation files found.")
    
    # Weather Data Tab
    with tab2:
        if weather_files:
            st.subheader("🌤️ Weather Data Analysis")
            
            # File selector
            selected_weather = st.selectbox(
                "Select Weather File",
                options=weather_files,
                format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
            )
            
            try:
                weather_df = pd.read_csv(selected_weather)
                
                # Weather metrics
                col1, col2, col3, col4 = st.columns(4)
                
                if 'temperature' in weather_df.columns:
                    with col1:
                        avg_temp = weather_df['temperature'].mean()
                        st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                
                if 'humidity' in weather_df.columns:
                    with col2:
                        avg_humidity = weather_df['humidity'].mean()
                        st.metric("Avg Humidity", f"{avg_humidity:.0f}%")
                
                if 'pressure' in weather_df.columns:
                    with col3:
                        avg_pressure = weather_df['pressure'].mean()
                        st.metric("Avg Pressure", f"{avg_pressure:.0f} hPa")
                
                if 'wind_speed' in weather_df.columns:
                    with col4:
                        avg_wind = weather_df['wind_speed'].mean()
                        st.metric("Avg Wind Speed", f"{avg_wind:.1f} m/s")
                
                st.markdown("---")
                
                # Temperature comparison
                if 'temperature' in weather_df.columns and 'location' in weather_df.columns:
                    st.subheader("🌡️ Temperature by Location")
                    
                    fig = px.bar(
                        weather_df,
                        x='location',
                        y='temperature',
                        title="Temperature Comparison Across Locations",
                        labels={'temperature': 'Temperature (°C)', 'location': 'Location'},
                        color='temperature',
                        color_continuous_scale='RdYlBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Weather conditions scatter
                if 'temperature' in weather_df.columns and 'humidity' in weather_df.columns and 'location' in weather_df.columns:
                    st.subheader("💧 Temperature vs Humidity")
                    
                    fig = px.scatter(
                        weather_df,
                        x='temperature',
                        y='humidity',
                        size='pressure' if 'pressure' in weather_df.columns else None,
                        color='location',
                        title="Weather Conditions by Location",
                        labels={'temperature': 'Temperature (°C)', 'humidity': 'Humidity (%)'},
                        hover_data=['location']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Raw weather data
                with st.expander("📋 View Weather Data"):
                    st.dataframe(weather_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading weather data: {str(e)}")
        else:
            st.info("No weather data files found.")
    
    # Combined Analysis Tab
    with tab3:
        st.subheader("📊 Combined Data Analysis")
        
        if ebird_files and weather_files:
            st.info("💡 Combined analysis will correlate bird observations with weather conditions.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### Available Analyses:
                - 🔗 Correlation between weather and bird activity
                - 📈 Species distribution vs temperature
                - 🌧️ Weather impact on observations
                - 🗺️ Geographic patterns with environmental factors
                """)
            
            with col2:
                st.markdown("""
                ### Coming Soon:
                - 🤖 ML-powered insights
                - 📊 Predictive analytics
                - 🎯 Stress level estimation
                - 📉 Trend analysis
                """)
            
            st.markdown("---")
            
            # Simple correlation if both datasets have compatible columns
            try:
                bird_df = pd.read_csv(ebird_files[0])
                weather_df = pd.read_csv(weather_files[0])
                
                st.success(f"✅ Loaded {len(bird_df)} bird observations and {len(weather_df)} weather records")
                
                # Show basic statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🐦 Bird Data Summary")
                    st.dataframe(bird_df.describe(), use_container_width=True)
                
                with col2:
                    st.markdown("#### 🌤️ Weather Data Summary")
                    st.dataframe(weather_df.describe(), use_container_width=True)
                
            except Exception as e:
                st.warning(f"⚠️ Could not perform combined analysis: {str(e)}")
        else:
            st.info("ℹ️ Collect both bird and weather data to enable combined analysis.")
