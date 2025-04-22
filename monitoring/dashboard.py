import streamlit as st
import pandas as pd
from monitoring.visualizer import plot_reconstruction_error

st.title("AI-Based Drift Detection Dashboard")

uploaded_file = st.file_uploader("Upload test CSV with drift", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())
    # Simulate drift detection (demo)
    error = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) ** 2
    errors = error.mean(axis=1)
    plot_reconstruction_error(errors, drift_points=[50, 120])
