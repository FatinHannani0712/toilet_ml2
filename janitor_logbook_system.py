
# ========================================
# FILE: janitor_logbook_system.py
# Description: Logging janitor action with ML prediction to Excel
# ========================================

import streamlit as st
import pandas as pd
import datetime
import os

st.title("🧹 Janitor Logbook System")

# Simulated ML prediction values (can be integrated later)
ammonia = st.number_input("Ammonia (ppm)", min_value=0.0, max_value=100.0, value=5.0)
iaq = st.number_input("Gas Resistance / IAQ (Ω)", min_value=0, max_value=100000, value=4000)
ml_prediction = st.selectbox("ML Prediction Level", ["Clean", "Normal", "Dirty", "Very Dirty"])

# Janitor Action Dropdown
action_type = st.selectbox("Select Janitor Action Type", [
    "Scheduled Shift Cleaning",
    "Demand Clean-Up",
    "Others"
])

if st.button("✅ Mark as Cleaned"):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    new_log = {
        "Date": date,
        "Time": time,
        "Ammonia": ammonia,
        "IAQ": iaq,
        "ML Prediction": ml_prediction,
        "Janitor Action Type": action_type
    }

    log_file = "janitor_log.xlsx"

    if os.path.exists(log_file):
        df_log = pd.read_excel(log_file)
        df_log = pd.concat([df_log, pd.DataFrame([new_log])], ignore_index=True)
    else:
        df_log = pd.DataFrame([new_log])

    df_log.to_excel(log_file, index=False)
    st.success(f"📝 Logged action successfully at {time}")

# View last 5 logs
if os.path.exists("janitor_log.xlsx"):
    st.subheader("📄 Recent Janitor Logs")
    df_recent = pd.read_excel("janitor_log.xlsx")
    st.dataframe(df_recent.tail(5))
