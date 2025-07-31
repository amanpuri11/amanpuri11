import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f5;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #2e7d32;
        font-size: 2.5em;
        text-align: center;
        padding: 20px 0;
    }
    .subtitle {
        color: #4b5e4b;
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    .stSlider > div > div {
        background-color: #a5d6a7;
        border-radius: 10px;
    }
    .predict-button {
        background-color: #388e3c;
        color: white;
        border-radius: 25px;
        padding: 10px 20px;
        font-size: 1.1em;
        text-align: center;
        display: block;
        margin: 20px auto;
        transition: background-color 0.3s;
    }
    .predict-button:hover {
        background-color: #2e7d32;
    }
    .result-box {
        background-color: #e8f5e9;
        border-left: 5px solid #388e3c;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 1.1em;
        color: #1b5e20;
    }
    .feature-box {
        background-color: #ffffff;
        border: 1px solid #dcedc8;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    label_encoder = LabelEncoder()
    df['Plant_Health_Status'] = label_encoder.fit_transform(df['Plant_Health_Status'])
    features = ['Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 'Humidity', 
                'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 'Phosphorus_Level', 
                'Potassium_Level', 'Chlorophyll_Content', 'Electrochemical_Signal']
    X = df[features]
    y = df['Plant_Health_Status']
    return X, y, label_encoder

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Streamlit app
def main():
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/48/000000/plant-under-sun.png")
        st.markdown("<h2 style='color: #388e3c;'>Plant Health Monitor</h2>", unsafe_allow_html=True)
        st.markdown("Use the sliders to input sensor data and predict plant health.")
        st.markdown("---")
        st.markdown("🌱 Built with Streamlit")

    # Main content
    st.markdown("<div class='title'>🌿 Plant Health Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Monitor your plant's health with real-time soil and micro-climate data</div>", unsafe_allow_html=True)

    # Load data and train model
    file_path = "plant_health_data.csv"
    X, y, label_encoder = load_and_preprocess_data(file_path)
    model, scaler = train_model(X, y)

    # Input form
    with st.container():
        st.markdown("### Enter Sensor Data")
        col1, col2 = st.columns(2)

        with col1:
            soil_moisture = st.slider("Soil Moisture (%)", 0.0, 50.0, 25.0, help="Percentage of water in the soil")
            ambient_temp = st.slider("Ambient Temperature (°C)", 0.0, 40.0, 20.0, help="Air temperature around the plant")
            soil_temp = st.slider("Soil Temperature (°C)", 0.0, 40.0, 20.0, help="Temperature of the soil")
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0, help="Relative humidity in the air")
            light_intensity = st.slider("Light Intensity (lux)", 0.0, 1000.0, 500.0, help="Light exposure level")
            soil_ph = st.slider("Soil pH", 0.0, 14.0, 7.0, help="Acidity or alkalinity of the soil")

        with col2:
            nitrogen_level = st.slider("Nitrogen Level (mg/kg)", 0.0, 50.0, 25.0, help="Nitrogen content in soil")
            phosphorus_level = st.slider("Phosphorus Level (mg/kg)", 0.0, 50.0, 25.0, help="Phosphorus content in soil")
            potassium_level = st.slider("Potassium Level (mg/kg)", 0.0, 50.0, 25.0, help="Potassium content in soil")
            chlorophyll_content = st.slider("Chlorophyll Content", 0.0, 50.0, 25.0, help="Measure of plant greenness")
            electrochemical_signal = st.slider("Electrochemical Signal", 0.0, 2.0, 1.0, help="Sensor signal strength")

        # Prepare input data for prediction
        input_data = np.array([[soil_moisture, ambient_temp, soil_temp, humidity, light_intensity, 
                                soil_ph, nitrogen_level, phosphorus_level, potassium_level, 
                                chlorophyll_content, electrochemical_signal]])
        input_data_scaled = scaler.transform(input_data)

        # Predict button
        if st.button("Predict Plant Health", key="predict", help="Click to predict plant health status"):
            prediction = model.predict(input_data_scaled)
            health_status = label_encoder.inverse_transform(prediction)[0]
            icon = "🌱" if health_status == "Healthy" else "⚠️" if health_status == "Moderate Stress" else "🚨"
            st.markdown(f"<div class='result-box'>{icon} Predicted Plant Health Status: <strong>{health_status}</strong></div>", unsafe_allow_html=True)

    # Feature importance (collapsible section)
    with st.expander("View Feature Importance"):
        st.markdown("### Feature Importance")
        feature_names = ['Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', 'Humidity', 
                         'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', 'Phosphorus_Level', 
                         'Potassium_Level', 'Chlorophyll_Content', 'Electrochemical_Signal']
        importance = model.feature_importances_
        for name, imp in zip(feature_names, importance):
            st.markdown(f"<div class='feature-box'>{name}: {imp:.4f}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()