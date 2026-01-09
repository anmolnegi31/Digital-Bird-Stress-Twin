"""
MLflow Dashboard page
"""

import streamlit as st

def show():
    st.title("📉 MLflow Dashboard")
    st.markdown("Track experiments and model performance.")
    
    st.info("🔗 Access MLflow UI at: http://localhost:5000")
    
    st.markdown("""
    ### Quick Access
    - **Experiments**: View all training runs
    - **Models**: Browse model registry
    - **Artifacts**: Download model checkpoints
    - **Metrics**: Compare model performance
    """)
    
    if st.button("🚀 Launch MLflow UI", type="primary"):
        st.code("mlflow ui --port 5000")
        st.info("Run the above command in your terminal to start MLflow UI")
    
    st.markdown("---")
    
    st.subheader("Recent Experiments")
    st.info("💡 Train a model to see experiments here!")
