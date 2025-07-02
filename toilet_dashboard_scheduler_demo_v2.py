
# ========================================
# FILE: toilet_dashboard_scheduler_demo_v2.py
# Description: Final Streamlit Dashboard for Toilet ML V2 with Hybrid Reasoning
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Toilet ML Dashboard V2", layout="wide")
st.sidebar.title("🚻 Toilet Monitoring Menu")
page = st.sidebar.radio("Go to:", [
    "1️⃣ Real Mode (Live Schedule)",
    "2️⃣ Real Mode - Manual Trigger",
    "3️⃣ Accelerated Schedule (Testing Mode)",
    "4️⃣ Demo Mode (Manual Input)",
    "5️⃣ Janitor Mode (Log & Actions)"
])

try:
    model_normal = load_model("cnn_lstm_model_normal.h5")
    model_spike = load_model("cnn_lstm_model_spike_short_seq.h5")
except:
    model_normal = None
    model_spike = None
    st.sidebar.warning("⚠️ Model not found. Please upload model files.")

if page == "1️⃣ Real Mode (Live Schedule)":
    st.header("📡 Real Mode - Live Monitoring")
    st.info("Coming soon: Live stream from IoT sensor + real-time ML prediction.")

elif page == "2️⃣ Real Mode - Manual Trigger":
    st.header("🔘 Real Mode - Manual Trigger")
    if st.button("📥 Load Latest Sensor Data"):
        st.success("✅ Simulated sensor data loaded!")
    if st.button("🚀 Run Hygiene Prediction"):
        st.warning("⚙️ Prediction logic coming soon...")

elif page == "3️⃣ Accelerated Schedule (Testing Mode)":
    st.header("⏩ Accelerated Testing Mode")
    uploaded_file = st.file_uploader("📂 Upload historical dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.success("✅ Data preview loaded.")

elif page == "4️⃣ Demo Mode (Manual Input)":
    st.header("🧪 Demo Mode - Manual Parameter Input")
    ammonia = st.slider("Ammonia (ppm)", 0.0, 50.0, 1.0)
    humidity = st.slider("Humidity (%)", 30.0, 100.0, 60.0)
    temperature = st.slider("Temperature (°C)", 20.0, 40.0, 30.0)
    iaq = st.slider("Gas Resistance / IAQ (Ω)", 0, 50000, 25000)
    visitor = st.slider("Daily Visitor Count", 0, 5000, 2000)
    hour = st.slider("Hour (0-23)", 0, 23, 12)

    features = np.array([[ammonia, humidity, temperature, iaq, hour, visitor]])
    features_scaled = features  # Add scaler if used in training

    st.subheader("🔍 Prediction Output:")
    if st.button("🚀 Predict Cleanliness"):
        prediction_text = ""
        confidence = 0.0

        if ammonia < 1.5 and iaq > 10000:
            if model_normal:
                result = model_normal.predict(features_scaled)
                label = np.argmax(result)
                confidence = float(np.max(result)) * 100
                final_label = ["Clean", "Normal"][label]
                prediction_text = f"🧼 Prediction: {final_label} ({confidence:.2f}%)"
        else:
            if model_spike:
                result = model_spike.predict(features_scaled.reshape(1, 5, 6))
                label = np.argmax(result)
                confidence = float(np.max(result)) * 100
                final_label = ["Dirty", "Very Dirty"][label]
                prediction_text = f"🚨 Prediction: {final_label} ({confidence:.2f}%)"

        st.success(prediction_text)

elif page == "5️⃣ Janitor Mode (Log & Actions)":
    st.header("🧹 Janitor Mode - Logging & Actions")
    if st.button("✅ Mark as Cleaned"):
        now = datetime.datetime.now()
        log = f"Cleaned at {now.strftime('%Y-%m-%d %H:%M:%S')}"
        st.success(log)
        with open("cleaning_log.txt", "a") as f:
            f.write(log + "\n")
    st.write("📄 Cleaning history:")
    if os.path.exists("cleaning_log.txt"):
        with open("cleaning_log.txt", "r") as f:
            logs = f.readlines()
        for line in logs[-5:]:
            st.text(line.strip())
    else:
        st.info("No cleaning logs yet.")
