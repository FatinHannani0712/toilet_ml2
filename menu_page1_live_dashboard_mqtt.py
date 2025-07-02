# ========================================
# FILE: menu_page1_live_dashboard_mqtt_debug.py
# Enhanced version with MQTT connection diagnostics
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

# ----------------------------
# MQTT Setup with Debugging
# ----------------------------
mqtt_broker = "cloud.lightsol.net"
mqtt_port = 1883
topics = {
    "ammonia": "esp32/BME680/PPM2",
    "temperature": "esp32/BME680/temperature2",
    "humidity": "esp32/BME680/humidity2",
    "iaq": "esp32/BME680/iaq2"
}

# Global variables for connection status
connection_status = {
    "connected": False,
    "last_message": None,
    "last_message_time": None,
    "error": None
}

latest_data = {"ammonia": None, "temperature": None, "humidity": None, "iaq": None}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        connection_status["connected"] = True
        connection_status["error"] = None
        for topic in topics.values():
            client.subscribe(topic, qos=1)
    else:
        connection_status["error"] = f"Connection failed with code {rc}"
        if rc == 1: connection_status["error"] += " (Incorrect protocol version)"
        elif rc == 2: connection_status["error"] += " (Invalid client identifier)"
        elif rc == 3: connection_status["error"] += " (Server unavailable)"
        elif rc == 4: connection_status["error"] += " (Bad username or password)"
        elif rc == 5: connection_status["error"] += " (Not authorized)"

def on_disconnect(client, userdata, rc):
    connection_status["connected"] = False
    if rc != 0:
        connection_status["error"] = f"Unexpected disconnection (rc: {rc})"
    # Attempt to reconnect
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
            except ValueError as e:
                latest_data[key] = None
                connection_status["error"] = f"Conversion error on {topic}: {str(e)}"

def on_log(client, userdata, level, buf):
    connection_status["last_log"] = buf

def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.on_log = on_log
    
    try:
        client.connect(mqtt_broker, mqtt_port, 60)
        client.loop_forever()
    except Exception as e:
        connection_status["error"] = f"Connection error: {str(e)}"

# Start MQTT thread only once
if 'mqtt_started' not in st.session_state:
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True
    time.sleep(1)  # Give it a moment to connect

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Live Toilet Hygiene Dashboard (MQTT Debug)", layout="wide")
st.title("🚻 Toilet Hygiene Live Dashboard (MQTT Debug Version)")

# Connection status panel
st.subheader("🔌 MQTT Connection Diagnostics")
col1, col2 = st.columns(2)
col1.metric("Connection Status", "Connected ✅" if connection_status["connected"] else "Disconnected ❌")
if connection_status["last_message_time"]:
    col2.metric("Last Message Received", connection_status["last_message_time"])

if connection_status["error"]:
    st.error(f"**Error:** {connection_status['error']}")

if connection_status["last_message"]:
    with st.expander("View Last Message Details"):
        st.code(connection_status["last_message"])

# Display raw connection info
with st.expander("Advanced Connection Info"):
    st.write("### MQTT Configuration")
    st.json({
        "broker": mqtt_broker,
        "port": mqtt_port,
        "topics": topics
    })
    
    st.write("### Latest Data Received")
    st.write(latest_data)
    
    st.write("### Thread Status")
    st.write(f"Active threads: {threading.active_count()}")
    st.write(f"MQTT thread alive: {'mqtt_started' in st.session_state}")

# ----------------------------
# Display Live Data
# ----------------------------
st.subheader("🟦 Live Sensor Readings (via MQTT)")
if all(value is not None for value in latest_data.values()):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ammonia (ppm)", f"{latest_data['ammonia']:.2f}")
    col2.metric("Temperature (°C)", f"{latest_data['temperature']:.2f}")
    col3.metric("Humidity (%)", f"{latest_data['humidity']:.2f}")
    col4.metric("IAQ (Ω)", f"{latest_data['iaq']:.2f}")

    # Bar chart
    fig = go.Figure(go.Bar(
        x=list(latest_data.keys()),
        y=list(latest_data.values()),
        marker_color=["blue", "red", "orange", "green"]
    ))
    fig.update_layout(title="Live Sensor Values", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Real-Time Rule-Based Hygiene
    # ----------------------------
    hygiene = "Clean"
    if latest_data['ammonia'] > 5 and (latest_data['humidity'] > 78 or latest_data['temperature'] > 26):
        hygiene = "Poor"
    st.success(f"🧼 Real-Time Hygiene Level: {hygiene}")

    # ----------------------------
    # Ammonia Risk
    # ----------------------------
    st.subheader("🟥 Ammonia Health Risk Score")
    risk = "Safe"
    if latest_data['ammonia'] > 8:
        risk = "⚠️ High Risk"
    elif latest_data['ammonia'] > 4:
        risk = "Moderate"
    st.info(f"☠️ Ammonia Risk Level: {risk}")

    # ----------------------------
    # ML Prediction
    # ----------------------------
    st.subheader("🟨 ML Prediction (Simulated)")
    try:
        model = load_model("cnn_lstm_model_normal.h5")
        df_input = pd.DataFrame([{
            "ammonia": latest_data['ammonia'],
            "humidity": latest_data['humidity'],
            "temperature": latest_data['temperature'],
            "iaq": latest_data['iaq'],
            "hour": datetime.datetime.now().hour,
            "daily_visitor": 80  # Placeholder
        }])
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df_input)
        X_input = np.expand_dims(df_scaled, axis=0)
        pred = model.predict(X_input)
        label = np.argmax(pred)
        label_name = ["Clean", "Normal"][label]
        confidence = float(np.max(pred)) * 100
        st.warning(f"📊 ML Prediction: {label_name} ({confidence:.2f}%)")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
else:
    st.warning("Waiting for complete MQTT data...")
    st.write("Current data state:")
    st.write(latest_data)

# Test connection button
if st.button("Force Reconnect to MQTT Broker"):
    if 'mqtt_started' in st.session_state:
        del st.session_state.mqtt_started
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True
    st.rerun()