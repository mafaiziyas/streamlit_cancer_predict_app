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
    # Resolves paths relative to your precise repository models folder layout
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
    
    # Standard min, max, and baseline defaults from the data distribution limits
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
    # Scales values manually between 0 and 1 using structural ranges for chart presentation symmetry
    scaled_dict = {}
    # Approximate maximums used as hard divisors for uniform radar scaling
    max_ranges = {
        "radius": 40.0, "texture": 50.0, "perimeter": 260.0, "area
