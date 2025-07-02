# Toilet Hygiene Forecasting with CNN + LSTM

# ============================
# STEP 1: DATA PREPROCESSING
# ============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import streamlit as st

# Load and clean dataset
df = pd.read_csv("merged_cleaned_dataset.csv")
df = df.dropna()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

features = ['ammonia', 'humidity', 'temperature', 'iaq', 'co2']
data = df[features].copy()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create time-series sequences
def create_sequences(data, input_steps=30, output_steps=60):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, 30, 60)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================
# STEP 2: MODELING (CNN + LSTM)
# ============================
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(30, 5)))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(60 * 5))  # output: 60 time steps * 5 features
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train.reshape((y_train.shape[0], -1)),
          epochs=20, batch_size=64, validation_split=0.1,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=1)

# ============================
# STEP 3: PERFORMANCE METRICS
# ============================
y_pred = model.predict(X_test)
y_pred_reshaped = y_pred.reshape((y_pred.shape[0], 60, 5))

mae = mean_absolute_error(y_test.reshape(-1, 5), y_pred_reshaped.reshape(-1, 5))
rmse = mean_squared_error(y_test.reshape(-1, 5), y_pred_reshaped.reshape(-1, 5), squared=False)
r2 = r2_score(y_test.reshape(-1, 5), y_pred_reshaped.reshape(-1, 5))

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# ============================
# STEP 4: STREAMLIT DASHBOARD
# ============================
# To run: `streamlit run toilet_cnn_lstm_model.py`

# st.title("Toilet Hygiene Forecast (Next 1 Hour)")
# input_window = data_scaled[-30:]
# input_window = input_window.reshape(1, 30, 5)
# prediction = model.predict(input_window).reshape(60, 5)
# prediction_df = pd.DataFrame(scaler.inverse_transform(prediction), columns=features)

# st.line_chart(prediction_df)
# st.dataframe(prediction_df)

# # Add rule-based logic
# latest_pred = prediction_df.iloc[-1]
# if latest_pred['ammonia'] > 5 and latest_pred['humidity'] > 80 and latest_pred['temperature'] > 26:
#     st.warning("Predicted Toilet Status: DIRTY")
# else:
#     st.success("Predicted Toilet Status: CLEAN")
