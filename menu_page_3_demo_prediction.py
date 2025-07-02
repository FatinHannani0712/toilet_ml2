import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("ml_cleanliness_model_v2.pkl")
scaler = joblib.load("scaler_cleanliness_v2.pkl")
performance = joblib.load("performance_cleanliness_v2.pkl")

st.title("🚻 Toilet Hygiene Prediction - DEMO Mode")
st.markdown("Fill in the latest sensor readings below:")

# --- User Input ---
ammonia = st.number_input("Ammonia (ppm)", min_value=0.0, step=0.1)
temp = st.number_input("Temperature (°C)", min_value=0.0, step=0.1)
humid = st.number_input("Humidity (%)", min_value=0.0, step=0.1)

# --- Predict Hygiene Level ---
if st.button("🔍 Predict Now"):
    # Preprocess input
    X = np.array([[ammonia, temp, humid]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    confidence = np.max(prob) * 100

    # Hygiene Mapping
    hygiene_labels = {0: "Clean", 1: "Normal", 2: "Dirty", 3: "Very Dirty"}
    hygiene_status = hygiene_labels.get(prediction, "Unknown")

    # --- IAQ Calculation ---
    iaq_score = temp + humid
    iaq_status = "Good" if iaq_score < 80 else "Moderate" if iaq_score < 100 else "Poor"

    # --- Event Trigger Check ---
    event_now = "✅ YES" if (ammonia > 5 or iaq_status == "Poor") else "❌ NO"

    # --- Display Output ---
    st.subheader("📊 Prediction Result")
    st.success(f"**Predicted Hygiene Level:** {hygiene_status} ({confidence:.2f}% confidence)")
    st.info(f"**Indoor Air Quality (IAQ):** {iaq_score:.2f} → {iaq_status}")
    st.warning(f"**Likely Dirty Event Happening Now?** {event_now}")

    # --- Model Performance ---
    st.subheader("📈 Model Performance")
    st.write("Accuracy:", performance["accuracy"])
    st.write("Precision:", performance["precision"])
    st.write("Recall:", performance["recall"])
    st.write("F1 Score:", performance["f1"])

    st.caption("This prediction is based on ML Toilet v2 model")
