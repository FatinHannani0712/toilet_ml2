# ========================================
# FILE: menu_page1_live_dashboard_mqtt_debug.py
# Final corrected version with proper refresh handling
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
from collections import deque

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
# Data Buffer for ML Model
# ----------------------------
DATA_BUFFER_SIZE = 10
data_buffer = deque(maxlen=DATA_BUFFER_SIZE)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        connection_status["connected"] = True
        connection_status["error"] = None
        for topic in topics.values():
            client.subscribe(topic, qos=1)
        st.session_state.last_update_time = time.time()
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
    try:
        connection_status["last_message"] = f"Topic: {msg.topic}, Payload: {msg.payload.decode()}"
        connection_status["last_message_time"] = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.last_update_time = time.time()

        # Update latest data
        for key, topic in topics.items():
            if msg.topic == topic:
                try:
                    if key == "pir":
                        latest_data[key] = int(msg.payload.decode())
                    else:
                        latest_data[key] = float(msg.payload.decode())
                except ValueError:
                    latest_data[key] = None

        # Update buffer when we have all required data
        if all(value is not None for value in latest_data.values()):
            new_entry = [
                latest_data['ammonia'],
                latest_data['humidity'],
                latest_data['temperature'],
                latest_data['iaq'],
                datetime.datetime.now().hour,
                80  # Placeholder for daily_visitor
            ]
            
            if len(data_buffer) == 0 or new_entry != data_buffer[-1]:
                data_buffer.append(new_entry)
                st.session_state.buffer_updated = True
                st.rerun()  # Changed from experimental_rerun

    except Exception as e:
        connection_status["error"] = f"Message processing error: {str(e)}"

def mqtt_thread():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    
    while True:
        try:
            client.connect(mqtt_broker, mqtt_port, 60)
            client.loop_forever()
        except Exception as e:
            connection_status["error"] = f"Connection error: {str(e)}"
            time.sleep(5)

# Initialize MQTT thread
if 'mqtt_started' not in st.session_state:
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True
    st.session_state.last_update_time = time.time()
    st.session_state.buffer_updated = False
    time.sleep(1)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Live Toilet Hygiene Dashboard (MQTT)", layout="wide")
st.title("🚻 Toilet Hygiene Live Dashboard (MQTT with New Sensors)")

# Auto-refresh configuration
auto_refresh = st.sidebar.checkbox("Enable Auto-refresh", True, help="Automatically refresh every 5 seconds")
if auto_refresh:
    refresh_placeholder = st.empty()
    for i in range(4, 0, -1):
        refresh_placeholder.write(f"Refreshing in {i} seconds...")
        time.sleep(1)
    st.rerun()  # Changed from experimental_rerun

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

# Data timeout check
if 'last_update_time' in st.session_state and (time.time() - st.session_state.last_update_time) > 30:
    st.warning("⚠️ No new data received for 30 seconds. Check MQTT connection.")

# ========================================
# MAIN DASHBOARD LAYOUT
# ========================================
st.title("🚻 LIVE TOILET HYGIENE MONITORING")
col_left, col_right = st.columns(2)

# LEFT COLUMN (LIVE SENSOR)
with col_left:
    st.subheader("🔴 REAL-TIME SENSOR DATA", divider="red")
    
    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.metric("**Ammonia**", 
                   f"{latest_data['ammonia']:.2f} ppm" if latest_data['ammonia'] is not None else "N/A", 
                   help="Danger level: >5ppm")
        col2.metric("**Temperature**", 
                   f"{latest_data['temperature']:.2f} °C" if latest_data['temperature'] is not None else "N/A", 
                   delta_color="off", help="Ideal: <26°C")
        
        col3, col4 = st.columns(2)
        col3.metric("**Humidity**", 
                   f"{latest_data['humidity']:.2f} %" if latest_data['humidity'] is not None else "N/A", 
                   delta_color="off", help="Ideal: <78%")
        col4.metric("**IAQ**", 
                   f"{latest_data['iaq']:.2f} Ω" if latest_data['iaq'] is not None else "N/A", 
                   help="Air quality")
        
        col5, col6 = st.columns(2)
        col5.metric("**CO₂**", 
                   f"{latest_data['co2']:.2f} ppm" if latest_data['co2'] is not None else "N/A")
        col6.metric("**PIR Motion**", 
                   "DETECTED" if latest_data['pir'] == 1 else "NO MOTION" if latest_data['pir'] == 0 else "N/A")

    st.subheader("🧼 HYGIENE STATUS", divider="blue")
    with st.container(border=True):
        if latest_data['ammonia'] is not None:
            ammonia_level = latest_data['ammonia']
            ammonia_progress = min(ammonia_level / 10 * 100, 100)
            st.markdown("**☠️ AMMONIA RISK LEVEL**")
            st.progress(
                int(ammonia_progress), 
                text=f"{ammonia_level:.2f} ppm ({'Safe' if ammonia_level < 4 else 'Danger'})"
            )
            
            hygiene_status = "POOR" if (ammonia_level > 5 and (
                latest_data['humidity'] > 78 or 
                latest_data['temperature'] > 26 or 
                latest_data['pir'] == 1
            )) else "CLEAN"
            
            st.markdown(f"**🧽 CURRENT STATUS:** <span style='color: {'red' if hygiene_status == 'POOR' else 'green'}; font-size: 20px'>{hygiene_status}</span>", 
                       unsafe_allow_html=True)
        else:
            st.warning("Waiting for ammonia data...")

# RIGHT COLUMN (ML PREDICTION)
with col_right:
    st.subheader("🟢 AI PREDICTION", divider="green")
    
    with st.container(border=True):
        st.markdown("**📌 Predict Value for the next 30 Minutes:**")
        col1, col2 = st.columns(2)
        col1.metric("Ammonia", f"{latest_data['ammonia']:.2f} ppm" if latest_data['ammonia'] is not None else "N/A")
        col2.metric("Temperature", f"{latest_data['temperature']:.2f} °C" if latest_data['temperature'] is not None else "N/A")
        
        col3, col4 = st.columns(2)
        col3.metric("Humidity", f"{latest_data['humidity']:.2f} %" if latest_data['humidity'] is not None else "N/A")
        col4.metric("IAQ", f"{latest_data['iaq']:.2f} Ω" if latest_data['iaq'] is not None else "N/A")
        
        try:
            if len(data_buffer) == DATA_BUFFER_SIZE:
                model = load_model("cnn_lstm_model_normal.h5")
                
                X_input = np.array(list(data_buffer))
                X_input = X_input.reshape(1, DATA_BUFFER_SIZE, 6)
                
                pred = model.predict(X_input)
                label = np.argmax(pred)
                label_name = ["CLEAN", "NORMAL", "DIRTY", "VERY DIRTY"][label]
                confidence = float(np.max(pred)) * 100
                
                st.markdown(f"**🤖 AI PREDICTION:** <span style='color: {'green' if label == 0 else 'orange' if label == 1 else 'red'}; font-size: 20px'>{label_name}</span>", 
                           unsafe_allow_html=True)
                st.metric("Confidence Level", f"{confidence:.2f}%")
                
                if label == 0:
                    st.success("✅ Toilet is clean!")
                elif label == 1:
                    st.warning("⚠️ Toilet needs routine cleaning")
                else:
                    st.error("❌ TOILET NEEDS IMMEDIATE CLEANING!")
            else:
                st.warning(f"Collecting data... ({len(data_buffer)}/{DATA_BUFFER_SIZE} samples)")
                
        except Exception as e:
            st.error(f"AI Error: {str(e)}")

# DEBUG SECTION
with st.expander("🔍 Debug Information"):
    st.write("Latest Data:", latest_data)
    st.write("Data Buffer Size:", len(data_buffer))
    if len(data_buffer) > 0:
        st.write("Latest Buffer Entry:", data_buffer[-1])
    st.write("Session State:", {k: v for k, v in st.session_state.items() if k != 'mqtt_client'})

# Reconnect button
if st.button("🔄 Reconnect MQTT"):
    if 'mqtt_started' in st.session_state:
        del st.session_state.mqtt_started
    data_buffer.clear()
    threading.Thread(target=mqtt_thread, daemon=True).start()
    st.session_state.mqtt_started = True
    st.rerun()  # Changed from experimental_rerun