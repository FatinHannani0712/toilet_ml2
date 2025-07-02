# ========================================
# FILE: menu_page1_live_dashboard_mqtt_debug.py
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
# Streamlit UI (Only ONE set_page_config here)
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

# ========================================
# BAHAGIAN UTAMA DASHBOARD (KIRI-KANAN)
# ========================================

st.title("🚻 LIVE TOILET HYGIENE MONITORING")

# Gunakan columns untuk bagi dua layout
col_left, col_right = st.columns(2)

# ================= KOLUM KIRI (LIVE SENSOR) =================
with col_left:
    st.subheader("🔴 REAL-TIME SENSOR DATA", divider="red")
    
    # Card untuk nilai sensor (lebih compact)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.metric("**Ammonia**", f"{latest_data['ammonia']:.2f} ppm", 
                   help="Level bahaya: >5ppm")
        col2.metric("**Temperature**", f"{latest_data['temperature']:.2f} °C", 
                   delta_color="off", help="Suhu ideal: <26°C")
        
        col3, col4 = st.columns(2)
        col3.metric("**Humidity**", f"{latest_data['humidity']:.2f} %", 
                   delta_color="off", help="Lembapan ideal: <78%")
        col4.metric("**IAQ**", f"{latest_data['iaq']:.2f} Ω", 
                   help="Kualiti udara")
        
        col5, col6 = st.columns(2)
        col5.metric("**CO₂**", f"{latest_data['co2']:.2f} ppm")
        col6.metric("**PIR Motion**", "DETECTED" if latest_data['pir'] == 1 else "NO MOTION")

    # Gantikan live graph dengan progress bar (lebih mudah dibaca janitor)
    st.subheader("🧼 HYGIENE STATUS", divider="blue")
    with st.container(border=True):
        # Progress bar untuk ammonia risk
        st.markdown("**☠️ AMMONIA RISK LEVEL**")
        ammonia_level = latest_data['ammonia']
        ammonia_progress = min(ammonia_level / 10 * 100, 100)  # Normalize to 0-10ppm scale
        st.progress(
            int(ammonia_progress), 
            text=f"{ammonia_level:.2f} ppm ({'Safe' if ammonia_level < 4 else 'Danger'})"
        )
        
        # Hygiene indicator (emoji + color)
        hygiene_status = "POOR" if (ammonia_level > 5 and (
            latest_data['humidity'] > 78 or 
            latest_data['temperature'] > 26 or 
            latest_data['pir'] == 1
        )) else "CLEAN"
        
        st.markdown(f"**🧽 CURRENT STATUS:** <span style='color: {'red' if hygiene_status == 'POOR' else 'green'}; font-size: 20px'>{hygiene_status}</span>", 
                   unsafe_allow_html=True)

# ================= KOLUM KANAN (ML PREDICTION) =================
with col_right:
    st.subheader("🟢 AI PREDICTION", divider="green")
    
    with st.container(border=True):
        # Paparkan nilai input model
        st.markdown("**📌 Predict Value for the next 30 Minutes:**")
        col1, col2 = st.columns(2)
        col1.metric("Ammonia", f"{latest_data['ammonia']:.2f} ppm" if latest_data['ammonia'] is not None else "N/A")
        col2.metric("Temperature", f"{latest_data['temperature']:.2f} °C" if latest_data['temperature'] is not None else "N/A")
        
        col3, col4 = st.columns(2)
        col3.metric("Humidity", f"{latest_data['humidity']:.2f} %" if latest_data['humidity'] is not None else "N/A")
        col4.metric("IAQ", f"{latest_data['iaq']:.2f} Ω" if latest_data['iaq'] is not None else "N/A")
        
        # Jalankan prediksi menggunakan method pertama
        try:
            if all(value is not None for value in latest_data.values()):
                model = load_model("cnn_lstm_model_normal.h5")
                
                # Sediakan data input sesuai method pertama
                input_data = {
                    "Ammonia (ppm)": latest_data['ammonia'],
                    "Humidity (%)": latest_data['humidity'],
                    "Temperature (°C)": latest_data['temperature'],
                    "IAQ (Ω)": latest_data['iaq'],
                    "Hour of Day": datetime.datetime.now().hour,
                    "Daily Visitor": 80  # placeholder
                }
                
                # Proses prediksi seperti method pertama
                X_input = np.array([list(input_data.values()) * DATA_BUFFER_SIZE])  # Shape: (1, 60)
                X_input = X_input.reshape(1, DATA_BUFFER_SIZE, 6)  # Reshape ke (1, 10, 6)
                
                pred = model.predict(X_input)
                label = np.argmax(pred)
                label_name = ["CLEAN", "NORMAL", "DIRTY", "VERY DIRTY"][label]
                confidence = float(np.max(pred)) * 100
                
                # Tampilkan hasil dengan styling dashboard kedua
                st.markdown(f"**🤖 AI PREDICTION:** <span style='color: {'green' if label == 0 else 'orange' if label == 1 else 'red'}; font-size: 20px'>{label_name}</span>", 
                           unsafe_allow_html=True)
                st.metric("Confidence Level", f"{confidence:.2f}%")
                
                # Tambah emoji berdasarkan hasil
                if label == 0:
                    st.success("✅ Toilet dalam keadaan bersih!")
                elif label == 1:
                    st.warning("⚠️ Toilet perlu pembersihan rutin")
                else:
                    st.error("❌ TOILET PERLU DIBERSIHKAN SEGERA!")
            else:
                st.warning("⚠️ Menunggu data lengkap dari semua sensor...")
                
        except Exception as e:
            st.error(f"AI Error: {str(e)}")

# ================= BAHAGIAN TAMBAHAN (WARNING UTK JANITOR) =================
if 'hygiene_status' in locals() and hygiene_status == "POOR":
    st.error("🚨 **ATTENTION JANITOR:** Toilet ini memerlukan pembersihan segera!", icon="⚠️")
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