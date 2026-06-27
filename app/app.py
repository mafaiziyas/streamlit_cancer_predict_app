import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# DATA AND ARTIFACT ACCESS LAYER 
@st.cache_resource
def load_analytical_pipeline():
    # Resolves paths relative to your precise repository models folder layout
    model_path = os.path.join(os.path.dirname(__file__), "../models/scaler_and_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

pipeline = load_analytical_pipeline()
scaler = pipeline["scaler"]
model = pipeline["model"]
feature_names = pipeline["feature_names"]

# CONTROL INTERFACE (DYNAMIC SIDEBAR SLIDERS) 
def add_sidebar():
    st.sidebar.header("🔬 Cell Nuclei Metrics")
    st.sidebar.markdown("Adjust parameters below to modify evaluation matrices manually.")
    
    # We maintain a robust map of slider limits to prevent out-of-bound evaluation parameters
    slider_labels = [
        ("Radius (mean)", "radius_mean", 0.0, 30.0, 14.0),
        ("Texture (mean)", "texture_mean", 0.0, 40.0, 19.0),
        ("Perimeter (mean)", "perimeter_mean", 0.0, 200.0, 92.0),
        ("Area (mean)", "area_mean", 0.0, 2500.0, 650.0),
        ("Smoothness (mean)", "smoothness_mean", 0.0, 0.2, 0.1),
        ("Compactness (mean)", "compactness_mean", 0.0, 0.4, 0.1),
        ("Concavity (mean)", "concavity_mean", 0.0, 0.5, 0.09),
        ("Concave points (mean)", "concave points_mean", 0.0, 0.2, 0.05),
        ("Symmetry (mean)", "symmetry_mean", 0.0, 0.4, 0.18),
        ("Fractal dimension (mean)", "fractal_dimension_mean", 0.0, 0.1, 0.06),
        ("Radius (se)", "radius_se", 0.0, 3.0, 0.4),
        ("Texture (se)", "texture_se", 0.0, 5.0, 1.2),
        ("Perimeter (se)", "perimeter_se", 0.0, 25.0, 2.9),
        ("Area (se)", "area_se", 0.0, 600.0, 40.0),
        ("Smoothness (se)", "smoothness_se", 0.0, 0.03, 0.007),
        ("Compactness (se)", "compactness_se", 0.0, 0.15, 0.025),
        ("Concavity (se)", "concavity_se", 0.0, 0.4, 0.03),
        ("Concave points (se)", "concave points_se", 0.0, 0.05, 0.01),
        ("Symmetry (se)", "symmetry_se", 0.0, 0.1, 0.02),
        ("Fractal dimension (se)", "fractal_dimension_se", 0.0, 0.03, 0.004),
        ("Radius (worst)", "radius_worst", 0.0, 40.0, 16.0),
        ("Texture (worst)", "texture_worst", 0.0, 50.0, 25.0),
        ("Perimeter (worst)", "perimeter_worst", 0.0, 260.0, 107.0),
        ("Area (worst)", "area_worst", 0.0, 4200.0, 880.0),
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

# MIN-MAX CUSTOM VECTOR LOCAL SCALE CALCULATOR 
def get_scaled_values(input_dict):
    # Dynamically scales values on a 0-1 baseline for visual standardization on the chart
    scaled_dict = {}
    for key, value in input_dict.items():
        # Using analytical safe ranges to establish clean scaling metrics
        scaled_dict[key] = value / (150.0 if "area" in key else 25.0 if "perimeter" in key else 1.0)
    return scaled_dict

# --- 4. DATA VISUALIZATION LAYER (POLAR RADAR CHART) ---
def get_radar_chart(input_data):
    scaled_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 
                  'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dim.']
    
    fig = go.Figure()

    # Emerald Green Theme for Core Means
    fig.add_trace(go.Scatterpolar(
        r=[scaled_data['radius_mean'], scaled_data['texture_mean'], scaled_data['perimeter_mean'],
           scaled_data['area_mean'], scaled_data['smoothness_mean'], scaled_data['compactness_mean'],
           scaled_data['concavity_mean'], scaled_data['concave points_mean'], scaled_data['symmetry_mean'],
           scaled_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='#2ecc71', width=2),
        name='Mean Attribute Value'
    ))
    
    # Soft Sky Blue Theme for Error Spreads
    fig.add_trace(go.Scatterpolar(
        r=[scaled_data['radius_se'], scaled_data['texture_se'], scaled_data['perimeter_se'],
           scaled_data['area_se'], scaled_data
