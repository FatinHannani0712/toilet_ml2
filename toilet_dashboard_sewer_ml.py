# toilet_dashboard_sewer_ml.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import requests

# Load Isolation Forest model
model = joblib.load("isolation_forest_sewer_model.pkl")

# Log file path
LOG_FILE = "sewer_event_log.csv"

# Telegram settings (replace with your own)
TELEGRAM_BOT_TOKEN = "<YOUR_BOT_TOKEN>"
TELEGRAM_CHAT_ID = "<YOUR_CHAT_ID>"
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# Streamlit UI
st.set_page_config(page_title="Sewer Event Monitor", layout="wide")
st.title("🚽 Sewer Event Detection Dashboard")

st.sidebar.header("Sensor Input")

# Input fields
sensor_input = {
    'ammonia': st.sidebar.number_input("Ammonia (ppm)", 0.0, 2000.0, step=1.0),
    'co2': st.sidebar.number_input("CO2", 0, 10000, step=50),
    'humidity': st.sidebar.number_input("Humidity (%)", 0.0, 100.0, step=0.1),
    'temperature': st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, step=0.1),
    'visitor': st.sidebar.number_input("Visitor Count", 0, 1000, step=1),
    'iaq': st.sidebar.number_input("IAQ Index", 0.0, 500.0, step=1.0),
    'delta_ammonia': st.sidebar.number_input("Delta Ammonia", 0.0, 200.0, step=0.1),
    'rolling_avg_ammonia': st.sidebar.number_input("Rolling Avg Ammonia", 0.0, 2000.0, step=1.0),
    'rolling_std_ammonia': st.sidebar.number_input("Rolling Std Ammonia", 0.0, 200.0, step=1.0),
    'IAQ_proxy': st.sidebar.number_input("IAQ Proxy (Temp + Humidity)", 0.0, 200.0, step=1.0),
    'hour': st.sidebar.slider("Hour of Day", 0, 23, 12)
}

# Process prediction
if sensor_input['ammonia'] > 100:
    input_df = pd.DataFrame([sensor_input])
    prediction = model.predict(input_df)[0]  # -1 = anomaly
    score = model.decision_function(input_df)[0]
    is_sewer = prediction == -1

    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Prediction:** {'🚨 Sewer Event' if is_sewer else '✅ Normal'}")
    st.markdown(f"**Anomaly Score:** {score:.4f}")

    st.markdown("### 📊 Sensor Readings")
    st.write(pd.DataFrame([sensor_input]))

    # Action taken
    if st.button("✅ Mark as Resolved"):
        log_entry = input_df.copy()
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry['prediction'] = 'Sewer Event' if is_sewer else 'Normal'
        log_entry['anomaly_score'] = score

        file_exists = os.path.exists(LOG_FILE)
        log_entry.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)

        # Telegram alert
        msg = f"✅ Sewer Event Resolved\nTimestamp: {log_entry['timestamp'].values[0]}\nAnomaly Score: {score:.4f}"
        sensor_lines = "\n".join([f"{k}: {v}" for k, v in sensor_input.items()])
        msg += f"\n\n📟 Sensor Readings:\n{sensor_lines}"

        if TELEGRAM_BOT_TOKEN != "<YOUR_BOT_TOKEN>":
            requests.post(TELEGRAM_API, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

        st.success("Sewer event marked as resolved and logged.")

else:
    st.info("Ammonia level too low to trigger sewer detection.")
