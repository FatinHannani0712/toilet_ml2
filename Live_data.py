# ========================================
# FILE: menu_page1_live_dashboard_mqtt_debug.py
# Updated for new sensors and ML with only 4 inputs
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
# MQTT Setup
# ----------------------------
# ========================================
# Updated MQTT Setup with IP Address
# ========================================

# ----------------------------
# MQTT Setup (using IP address)
# ----------------------------
mqtt_broker = "134.209.100.187"  # Changed from "cloud.lightsol.net"
mqtt_port = 1883  # Same as before

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

    # ----------------------------
    # ML Prediction (Only Using Original 4 Sensors)
    # ----------------------------
    st.subheader("🟨 ML Prediction (Based on Original Features Only)")
    # Corrected version:
try:
    model = load_model("cnn_lstm_model_normal.h5")
    df_input = pd.DataFrame([{
        "ammonia": latest_data['ammonia'],
        "humidity": latest_data['humidity'],
        "temperature": latest_data['temperature'],
        "iaq": latest_data['iaq'],
        "hour": datetime.datetime.now().hour,
        "daily_visitor": 80
    }])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_input)
    X_input = np.expand_dims(df_scaled, axis=0)
    pred = model.predict(X_input)
    label = np.argmax(pred)
    label_name = ["Clean", "Normal", "Dirty", "Very Dirty"][label]
    confidence = float(np.max(pred)) * 100
    st.warning(f"📊 ML Prediction: {label_name} ({confidence:.2f}%)")
except Exception as e:
    st.error(f"Prediction Error: {str(e)}")

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
