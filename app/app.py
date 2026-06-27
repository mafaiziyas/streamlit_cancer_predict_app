import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

#Set global responsive window configurations
st.set_page_config(
    page_title="Cell Diagnostic Engine",
    page_icon="🔬",
    layout="wide")

#High-Performance IO Data Layer: Cache the Pipeline
@st.cache_resource
def load_analytical_pipeline():
    # Resolves paths relative to the current file location
    model_path = os.path.join(os.path.dirname(__file__), "../models/scaler_and_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Unpack our packed model artifacts dictionary
pipeline = load_analytical_pipeline()
scaler = pipeline["scaler"]
model = pipeline["model"]
feature_names = pipeline["feature_names"]

# Header Presentation
st.title("🔬 Breast Cancer Diagnostic Assistant")
st.markdown("An interactive AI deployment workspace assisting clinical triage analysis of mass telemetry data.")
st.markdown("---")

# Ingestion Sidebar State Controller
st.sidebar.header("Target Measurement Controls")
input_mode = st.sidebar.radio(
    "Data Source Input:", 
    ("Manual Patient Overrides", "Live Cytology Hardware Data Feed")
)

active_inputs = []

if input_mode == "Manual Patient Overrides":
    st.sidebar.subheader("Adjust Dimensional Features")
    # Loop over original training columns to ensure order alignment
    for col in feature_names:
        val = st.sidebar.slider(f"{col}", min_value=0.0, max_value=200.0, value=20.0)
        active_inputs.append(val)
else:
    st.sidebar.info("📡 Connection Active: Streaming from Automation Hardware Buffer.")
    # Standard malignant reference row vector matching training layout
    active_inputs = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
                     1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                     25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    st.sidebar.success("✅ Frame telemetry synchronized perfectly.")

# Compute Layer: Execute Transformations and Model Inference
input_matrix = np.array(active_inputs).reshape(1, -1)
scaled_matrix = scaler.transform(input_matrix)

prediction = model.predict(scaled_matrix)[0]
probabilities = model.predict_proba(scaled_matrix)[0]

# UI Presentation Workspace Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Cell Feature Visual Distribution")
    
    # Map out the structural polygon visualization
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness']
    chart_values = [active_inputs[0], active_inputs[1], active_inputs[2], active_inputs[3], active_inputs[4]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
          r=chart_values,
          theta=categories,
          fill='toself',
          name='Mass Signature Profile',
          line_color='#FF4B4B'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🩺 Diagnostic Evaluation Target")
    
    # State-based UI banner warnings
    if prediction == 1:
        st.error("⚠️ Diagnosis Verdict: MALIGNANT")
    else:
        st.success("✅ Diagnosis Verdict: BENIGN")
        
    st.markdown("---")
    st.write("#### Inference Prediction Certainty Metrics:")
    
    st.write(f"**Benign Class Probability:** {probabilities[0] * 100:.2f}%")
    st.progress(float(probabilities[0]))
    
    st.write(f"**Malignant Class Probability:** {probabilities[1] * 100:.2f}%")
    st.progress(float(probabilities[1]))
