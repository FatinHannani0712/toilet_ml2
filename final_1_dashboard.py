# FINAL DASHBOARD: final1_dashboard.py
# Combines real-time MQTT + ML prediction (CNN-LSTM)

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import threading
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import paho.mqtt.client as mqtt

# MQTT Setup
mqtt_broker = "134.209.100.187"
mqtt_port = 1883
topics = {
    "ammonia": "esp32/BME680/PPM1",
    "temperature": "esp32/BME680/temperature1",
    "humidity": "esp32/BME680/humidity1",
    "iaq": "esp32/BME680/iaq1",
    "co2": "esp32/BME680/co21",
    "pir": "esp32/BME680/PIR1"
}

latest_data = {k: None for k in topics.keys()}
connection_status = {"connected": False, "last_msg": None, "last_time": None, "error": None}

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        connection_status["connected"] = True
        for t in topics.values():
            client.subscribe(t)

def on_message(client, userdata, msg):
    connection_status["last_msg"] = f"{msg.topic}: {msg.payload.decode()}"
    connection_status["last_time"] = datetime.datetime.now().strftime("%H:%M:%S")
    for k, t in topics.items():
        if msg.topic == t:
            try:
                latest_data[k] = float(msg.payload.decode()) if k != "pir" else int(msg.payload.decode())
            except:
                pass

def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(mqtt_broker, mqtt_port, 60)
    client.loop_forever()

if 'mqtt_started' not in st.session_state:
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True

# Streamlit UI
st.set_page_config(page_title="Toilet Hygiene Final Dashboard", layout="wide")
st.title("🚻 Toilet Hygiene Final Dashboard")

st.markdown(f"**Auto-refresh** every 5s | **Last MQTT**: {connection_status['last_time'] or 'Waiting...'}")

# Live sensor display
if all(v is not None for v in latest_data.values()):
    st.subheader("🔴 Real-Time Sensor Values")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Ammonia (ppm)", f"{latest_data['ammonia']:.2f}")
    col2.metric("Temperature (°C)", f"{latest_data['temperature']:.2f}")
    col3.metric("Humidity (%)", f"{latest_data['humidity']:.2f}")
    col4.metric("IAQ (Ω)", f"{latest_data['iaq']:.2f}")
    col5.metric("CO₂ (ppm)", f"{latest_data['co2']:.2f}")
    col6.metric("PIR", "Detected" if latest_data['pir'] else "None")

    st.plotly_chart(go.Figure(data=[
        go.Bar(x=list(topics.keys())[:-1], y=[
            latest_data['ammonia'], latest_data['temperature'],
            latest_data['humidity'], latest_data['iaq'], latest_data['co2']
        ])
    ]), use_container_width=True)

    # Hygiene logic
    st.subheader("🧼 Hygiene Logic Status")
    def classify(a, h, t):
        if a > 1.5 and h > 50 and t > 27:
            return "High Hygiene Risk"
        elif a > 1.0 and h > 45 and t > 26:
            return "Medium Hygiene Risk"
        else:
            return "Low Hygiene Risk"
    logic_status = classify(latest_data['ammonia'], latest_data['humidity'], latest_data['temperature'])
    st.info(f"Predicted Hygiene Level (Logic): **{logic_status}**")

    # ML prediction
    st.subheader("🤖 ML-Based Sensor Forecast (Next 1 Hour)")
    try:
        model = load_model("cnn_lstm_forecast_1hour.h5")
        scaler = MinMaxScaler()
        df_input = pd.DataFrame([{
            "ammonia": latest_data['ammonia'],
            "humidity": latest_data['humidity'],
            "temperature": latest_data['temperature'],
            "iaq": latest_data['iaq'],
            "co2": latest_data['co2']
        }])
        scaled = scaler.fit_transform(df_input)
        window = np.repeat(scaled, 30, axis=0).reshape(1, 30, 5)
        pred = model.predict(window).reshape(60, 5)
        pred_df = pd.DataFrame(scaler.inverse_transform(pred), columns=["ammonia", "humidity", "temperature", "iaq", "co2"])

        st.line_chart(pred_df)
        st.dataframe(pred_df)

        final_pred = pred_df.iloc[-1]
        logic_pred = classify(final_pred['ammonia'], final_pred['humidity'], final_pred['temperature'])
        st.success(f"ML-Predicted Hygiene Level (in 1 Hour): **{logic_pred}**")

    except Exception as e:
        st.error(f"ML Prediction Error: {str(e)}")
else:
    st.warning("⏳ Waiting for full sensor data... please wait.")
