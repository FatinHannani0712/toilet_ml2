# ========================================
#Toilet Hygiene Live Dashboard (MQTT with New Sensors)
# Updated with data buffer for CNN-LSTM model
# ========================================

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
from collections import deque  # <-- New import for data buffer

# ----------------------------
# MQTT Setup
# ----------------------------
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

connection_status = {
    "connected": False,
    "last_message": None,
    "last_message_time": None,
    "error": None
}

latest_data = {
    "ammonia": None,
    "temperature": None,
    "humidity": None,
    "iaq": None,
    "co2": None,
    "pir": None
}

# ----------------------------
# Data Buffer for ML Model (NEW)
# ----------------------------
DATA_BUFFER_SIZE = 10  # Must match model's expected timesteps
data_buffer = deque(maxlen=DATA_BUFFER_SIZE)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        connection_status["connected"] = True
        connection_status["error"] = None
        for topic in topics.values():
            client.subscribe(topic, qos=1)
    else:
        connection_status["error"] = f"Connection failed with code {rc}"

def on_disconnect(client, userdata, rc):
    connection_status["connected"] = False
    if rc != 0:
        connection_status["error"] = f"Unexpected disconnection (rc: {rc})"
    try:
        client.reconnect()
    except Exception as e:
        connection_status["error"] = f"Reconnection failed: {str(e)}"

def on_message(client, userdata, msg):
    connection_status["last_message"] = f"Topic: {msg.topic}, Payload: {msg.payload.decode()}"
    connection_status["last_message_time"] = datetime.datetime.now().strftime("%H:%M:%S")

    for key, topic in topics.items():
        if msg.topic == topic:
            try:
                latest_data[key] = float(msg.payload.decode())
            except ValueError:
                if key == "pir":
                    try:
                        latest_data[key] = int(msg.payload.decode())
                    except:
                        latest_data[key] = None
                else:
                    latest_data[key] = None
    
    # Update data buffer when new message arrives (NEW)
    if all(value is not None for value in latest_data.values()):
        data_buffer.append([
            latest_data['ammonia'],
            latest_data['humidity'],
            latest_data['temperature'],
            latest_data['iaq'],
            datetime.datetime.now().hour,
            80  # Placeholder for daily_visitor
        ])
        st.write("🔄 Data baru ditambahkan ke buffer:", list(data_buffer))  # <-- DEBUG LINE

def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    try:
        client.connect(mqtt_broker, mqtt_port, 60)
        client.loop_forever()
    except Exception as e:
        connection_status["error"] = f"Connection error: {str(e)}"

if 'mqtt_started' not in st.session_state:
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True
    time.sleep(1)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Live Toilet Hygiene Dashboard (MQTT)", layout="wide")
st.title("🚻 Toilet Hygiene Live Dashboard (MQTT with New Sensors)")

# Diagnostics
st.subheader("🔌 MQTT Connection")
col1, col2 = st.columns(2)
col1.metric("Connection", "✅ Connected" if connection_status["connected"] else "❌ Disconnected")
if connection_status["last_message_time"]:
    col2.metric("Last Message", connection_status["last_message_time"])
if connection_status["error"]:
    st.error(f"⚠️ {connection_status['error']}")
if connection_status["last_message"]:
    with st.expander("Last MQTT Message"):
        st.code(connection_status["last_message"])

# MQTT config details
with st.expander("📡 MQTT Configuration"):
    st.json({
        "broker": mqtt_broker,
        "port": mqtt_port,
        "topics": topics
    })

# ----------------------------
# Live Sensor Data
# ----------------------------
st.subheader("🟦 Live Sensor Readings")
if all(value is not None for value in latest_data.values()):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Ammonia (ppm)", f"{latest_data['ammonia']:.2f}")
    col2.metric("Temperature (°C)", f"{latest_data['temperature']:.2f}")
    col3.metric("Humidity (%)", f"{latest_data['humidity']:.2f}")
    col4.metric("IAQ (Ω)", f"{latest_data['iaq']:.2f}")
    col5.metric("CO₂ (ppm)", f"{latest_data['co2']:.2f}")
    pir_status = "Detected" if latest_data['pir'] == 1 else "None"
    col6.metric("PIR Motion", pir_status)

    # Bar chart
    fig = go.Figure(go.Bar(
        x=["Ammonia", "Temperature", "Humidity", "IAQ", "CO2"],
        y=[
            latest_data['ammonia'],
            latest_data['temperature'],
            latest_data['humidity'],
            latest_data['iaq'],
            latest_data['co2']
        ],
        marker_color=["blue", "red", "orange", "green", "purple"]
    ))
    fig.update_layout(title="Live Sensor Values", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Hygiene Logic
    # ----------------------------
    st.subheader("🧽 Hygiene Level (Real-Time)")
    hygiene = "Clean"
    if latest_data['ammonia'] > 5 and (
        latest_data['humidity'] > 78 or
        latest_data['temperature'] > 26 or
        latest_data['pir'] == 1
    ):
        hygiene = "Poor"
    st.success(f"🧼 Hygiene Level: {hygiene}")

    # ----------------------------
    # Ammonia Risk
    # ----------------------------
    st.subheader("🟥 Ammonia Health Risk")
    risk = "Safe"
    if latest_data['ammonia'] > 8:
        risk = "⚠️ High Risk"
    elif latest_data['ammonia'] > 4:
        risk = "Moderate"
    st.info(f"☠️ Ammonia Risk Level: {risk}")

# ========================================
# BAHAGIAN ML PREDICTION (REAL-TIME)
# ========================================

st.subheader("🟨 ML Prediction (Real-Time Instant)")
try:
    if all(value is not None for value in latest_data.values()):
        # Gunakan data aktual + generate data historis yang realistic
        current_hour = datetime.datetime.now().hour
        
        # Buat data input dengan variasi kecil (simulasi data historis)
        X_input = np.array([
            [latest_data['ammonia'] * (0.95 + 0.1*i/DATA_BUFFER_SIZE),
             latest_data['humidity'] * (0.97 + 0.06*i/DATA_BUFFER_SIZE),
             latest_data['temperature'],
             latest_data['iaq'],
             current_hour,
             80]  # visitor placeholder
            for i in range(DATA_BUFFER_SIZE)
        ])
        
        # Reshape untuk model (1, timesteps, features)
        X_input = X_input.reshape(1, DATA_BUFFER_SIZE, 6)
        
        # Prediksi
        pred = model.predict(X_input)
        label = np.argmax(pred)
        label_name = ["Bersih", "Normal", "Kotor", "Sangat Kotor"][label]
        confidence = float(np.max(pred)) * 100
        
        # Tampilkan hasil
        st.warning(f"📊 **Prediksi ML:** {label_name} ({confidence:.2f}%)")
        
        # Visualisasi trend simulasi
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=X_input[0,:,0], 
            name="Ammonia (simulasi)",
            line=dict(color='red')))
        fig.update_layout(title="Simulasi Trend Data untuk Prediksi")
        st.plotly_chart(fig)
        
    else:
        st.warning("⚠️ Data sensor belum lengkap...")
        
except Exception as e:
    st.error(f"Error prediksi: {str(e)}")

else:
    st.warning("⚠️ Waiting for complete data from all sensors...")
    st.write("Current sensor state:")
    st.write(latest_data)

# Reconnect button
if st.button("🔄 Reconnect MQTT"):
    if 'mqtt_started' in st.session_state:
        del st.session_state.mqtt_started
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True
    st.rerun()