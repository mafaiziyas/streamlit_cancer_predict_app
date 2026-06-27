import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

# 1. CACHED PIPELINE RECOVERY LAYER
@st.cache_resource
def load_analytical_pipeline():
    # Resolves paths relative to your precise repository layout
    model_path = os.path.join(os.path.dirname(__file__), "../models/scaler_and_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

pipeline = load_analytical_pipeline()
scaler = pipeline["scaler"]
model = pipeline["model"]
feature_names = pipeline["feature_names"]


# 2. DATA INGESTION AND CLEANING LAYER
@st.cache_data
def get_clean_data():
    # Construct columns dynamically using 'id', 'diagnosis', followed by the 30 trained features
    column_names = ['id', 'diagnosis'] + list(feature_names)
    
    data = pd.read_csv("data/wdbc.data", header=None, names=column_names)
    data = data.drop(['id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


# 3. INTERACTIVE CONTROL SIDEBAR
def add_sidebar():
    st.sidebar.header("🔬 Cell Nuclei Measurements")
    st.sidebar.markdown("Adjust parameters manually to update evaluation matrices.")
    
    data = get_clean_data()
    input_dict = {}

    # Dynamically generate sliders matching your exact feature array configuration
    for key in feature_names:
        # Create user-friendly labels by replacing underscores with clean spaces
        clean_label = key.replace("_", " ").title()
        
        input_dict[key] = st.sidebar.slider(
            clean_label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
        
    return input_dict


# 4. MIN-MAX SCALER FOR THE RADAR VISUALIZATION ONLY
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        # Prevent division by zero if limits collapse
        scaled_dict[key] = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.0
        
    return scaled_dict


# 5. HIGH-DIMENSIONAL RADAR SPECTRUM PLOT
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    
    # Unified categorical groupings for the 10 core dimensions
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 
                  'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    # Trace 1: Mean Attributes (Translucent Royal Blue)
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave_points_mean'], input_data['symmetry_mean'],
           input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        fillcolor='rgba(41, 128, 185, 0.2)',
        line=dict(color='#2980b9', width=2),
        name='Mean Value'
    ))
    
    # Trace 2: Standard Errors (Translucent Emerald Green)
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
           input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
           input_data['concave_points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.2)',
        line=dict(color='#2ecc71', width=2),
        name='Standard Error'
    ))
    
    # Trace 3: Worst Case Boundary (Translucent Coral Red)
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave_points_worst'], input_data['symmetry_worst'],
           input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.15)',
        line=dict(color='#e74c3c', width=2),
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig


# 6. INFERENCE PRODUCTION DIAGNOSTICS DISPLAY PANEL
def add_predictions(input_data):
    # Ensure features match the exact order expected by your pipeline's training matrix
    ordered_values = [input_data[name] for name in feature_names]
    
    input_array = np.array(ordered_values).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)[0]
    probabilities = model.predict_proba(input_array_scaled)[0]
    
    st.subheader("🩺 Cell Cluster Prediction")
    st.write("The current mathematical assessment flags this cluster as:")
    
    if prediction == 0:
        st.success("Verdict: BENIGN")
    else:
        st.error("Verdict: MALIGNANT")
        
    st.markdown("---")
    st.write(f"**Probability of being Benign:** {probabilities[0] * 100:.2f}%")
    st.progress(float(probabilities[0]))
    
    st.write(f"**Probability of being Malignant:** {probabilities[1] * 100:.2f}%")
    st.progress(float(probabilities[1]))
    
    st.caption("ℹ️ *This system acts as a pipeline helper tool and shouldn't replace certified diagnostic guidance.*")


# 7. MAIN INTERFACE ORCHESTRATOR
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Capture dynamic slider values
    input_data = add_sidebar()
    
    with st.container():
        st.title("🔬 Breast Cancer Diagnostic Assistant")
        st.write("This workspace runs diagnostic inferences predicting whether a sample tissue mass is benign or malignant. Update metrics manually using the sliders on the left control dock.")
        st.markdown("---")
        
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📊 Cell Feature Profile Matrix")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
        
    with col2:
        add_predictions(input_data)


if __name__ == '__main__':
    main()
