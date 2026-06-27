%%writefile app/app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# 1. APPLICATION VIEWCONFIG LAYOUT
st.set_page_config(
    page_title="Breast Cancer Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CACHED SERIALIZED WEIGHTS ENGINE RECOVERY
@st.cache_resource
def load_analytical_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "../models/scaler_and_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

pipeline = load_analytical_pipeline()
scaler = pipeline["scaler"]
model = pipeline["model"]
feature_names = pipeline["feature_names"]

# 3. SIDEBAR CONTROLS GENERATION LAYER
def add_sidebar():
    st.sidebar.header("🔬 Cell Nuclei Metrics")
    st.sidebar.markdown("Adjust parameters below to modify evaluation matrices manually.")
    
    slider_labels = [
        ("Radius (mean)", "radius_mean", 0.0, 30.0, 14.1),
        ("Texture (mean)", "texture_mean", 0.0, 40.0, 19.3),
        ("Perimeter (mean)", "perimeter_mean", 0.0, 200.0, 92.0),
        ("Area (mean)", "area_mean", 0.0, 2500.0, 654.8),
        ("Smoothness (mean)", "smoothness_mean", 0.0, 0.2, 0.1),
        ("Compactness (mean)", "compactness_mean", 0.0, 0.4, 0.1),
        ("Concavity (mean)", "concavity_mean", 0.0, 0.5, 0.09),
        ("Concave points (mean)", "concave points_mean", 0.0, 0.2, 0.05),
        ("Symmetry (mean)", "symmetry_mean", 0.0, 0.4, 0.18),
        ("Fractal dimension (mean)", "fractal_dimension_mean", 0.0, 0.1, 0.06),
        ("Radius (se)", "radius_se", 0.0, 3.0, 0.4),
        ("Texture (se)", "texture_se", 0.0, 5.0, 1.2),
        ("Perimeter (se)", "perimeter_se", 0.0, 25.0, 2.9),
        ("Area (se)", "area_se", 0.0, 600.0, 40.3),
        ("Smoothness (se)", "smoothness_se", 0.0, 0.03, 0.007),
        ("Compactness (se)", "compactness_se", 0.0, 0.15, 0.025),
        ("Concavity (se)", "concavity_se", 0.0, 0.4, 0.03),
        ("Concave points (se)", "concave points_se", 0.0, 0.05, 0.01),
        ("Symmetry (se)", "symmetry_se", 0.0, 0.1, 0.02),
        ("Fractal dimension (se)", "fractal_dimension_se", 0.0, 0.03, 0.004),
        ("Radius (worst)", "radius_worst", 0.0, 40.0, 16.3),
        ("Texture (worst)", "texture_worst", 0.0, 50.0, 25.7),
        ("Perimeter (worst)", "perimeter_worst", 0.0, 260.0, 107.2),
        ("Area (worst)", "area_worst", 0.0, 4200.0, 880.6),
        ("Smoothness (worst)", "smoothness_worst", 0.0, 0.3, 0.13),
        ("Compactness (worst)", "compactness_worst", 0.0, 1.1, 0.25),
        ("Concavity (worst)", "concavity_worst", 0.0, 1.3, 0.27),
        ("Concave points (worst)", "concave points_worst", 0.0, 0.3, 0.11),
        ("Symmetry (worst)", "symmetry_worst", 0.0, 0.7, 0.29),
        ("Fractal dimension (worst)", "fractal_dimension_worst", 0.0, 0.2, 0.08)
    ]
    
    input_dict = {}
    for label, key, min_v, max_v, default_v in slider_labels:
        input_dict[key] = st.sidebar.slider(label, min_value=min_v, max_value=max_v, value=default_v)
        
    return input_dict

# 4. CHANNELS SCALER MATRIX FOR RADAR VISUAL BALANCE
def get_visual_scaled_values(input_dict):
    scaled_dict = {}
    max_ranges = {
        "radius": 40.0, "texture": 50.0, "perimeter": 260.0, "area": 4200.0,
        "smoothness": 0.3, "compactness": 1.1, "concavity": 1.3, "concave points": 0.3,
        "symmetry": 0.7, "fractal_dimension": 0.2
    }
    
    for key, value in input_dict.items():
        base_metric = next((m for m in max_ranges if m in key.replace("_", " ")), None)
        if base_metric:
            scaled_dict[key] = value / max_ranges[base_metric]
        else:
            scaled_dict[key] = value
            
    return scaled_dict

# 5. RADAR PROFILE GENERATION
def get_radar_chart(input_data):
    scaled = get_visual_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 
                  'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dim.']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[scaled['radius_mean'], scaled['texture_mean'], scaled['perimeter_mean'], scaled['area_mean'], 
           scaled['smoothness_mean'], scaled['compactness_mean'], scaled['concavity_mean'], 
           scaled['concave points_mean'], scaled['symmetry_mean'], scaled['fractal_dimension_mean']],
        theta=categories, fill='toself', fillcolor='rgba(41, 128, 185, 0.2)',
        line=dict(color='#2980b9', width=2), name='Mean Attributes'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[scaled['radius_se'], scaled['texture_se'], scaled['perimeter_se'], scaled['area_se'], 
           scaled['smoothness_se'], scaled['compactness_se'], scaled['concavity_se'], 
           scaled['concave points_se'], scaled['symmetry_se'], scaled['fractal_dimension_se']],
        theta=categories, fill='toself', fillcolor='rgba(26, 188, 156, 0.2)',
        line=dict(color='#1abc9c', width=2), name='Standard Error'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[scaled['radius_worst'], scaled['texture_worst'], scaled['perimeter_worst'], scaled['area_worst'], 
           scaled['smoothness_worst'], scaled['compactness_worst'], scaled['concavity_worst'], 
           scaled['concave points_worst'], scaled['symmetry_worst'], scaled['fractal_dimension_worst']],
        theta=categories, fill='toself', fillcolor='rgba(231, 76, 60, 0.15)',
        line=dict(color='#e74c3c', width=2), name='Worst Boundary Case'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(180, 180, 180, 0.3)"),
            angularaxis=dict(gridcolor="rgba(180, 180, 180, 0.3)")
        ),
        showlegend=True
    )
    return fig

# 6. INFERENCE EVALUATION METRIC DISPLAY PANEL
def display_predictions(input_data):
    ordered_values = [input_data[name] for name in feature_names]
    input_array = np.array(ordered_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    st.subheader("🩺 Diagnostic Classification Analysis")
    
    if prediction == 0:
        st.success("Verdict Status: BENIGN")
    else:
        st.error("Verdict Status: MALIGNANT")
        
    st.markdown("---")
    st.write(f"**Benign Probability Index:** {probabilities[0] * 100:.2f}%")
    st.progress(float(probabilities[0]))
    st.write(f"**Malignant Probability Index:** {probabilities[1] * 100:.2f}%")
    st.progress(float(probabilities[1]))

# 7. MAIN ORCHESTRATION PIPELINE
def main():
    st.title("🔬 Breast Cancer Diagnostic Assistant")
    st.markdown("---")
    input_data = add_sidebar()
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📊 Cell Profiling Radar Spectrum Matrix")
        st.plotly_chart(get_radar_chart(input_data), use_container_width=True)
    with col2:
        display_predictions(input_data)

if __name__ == '__main__':
    main()
