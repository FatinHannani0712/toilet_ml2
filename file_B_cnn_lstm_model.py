
# ===================================================
# FILE B: Label Creation + CNN-LSTM Model for Toilet Cleanliness Classification
# ===================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# -----------------------------
# Load cleaned data
# -----------------------------
df = pd.read_csv("cleaned_data_for_ml.csv")

# -----------------------------
# Generate Cleanliness Labels
# -----------------------------
def classify_cleanliness(row):
    if row['ammonia'] <= 0.5 and row['iaq'] >= 30000:
        return "Clean"
    elif row['ammonia'] <= 1.5:
        return "Normal"
    elif row['ammonia'] <= 4.0 or row['iaq'] < 5000:
        return "Dirty"
    else:
        return "Very Dirty"

df['cleanliness_level'] = df.apply(classify_cleanliness, axis=1)

# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['cleanliness_level'])
label_names = label_encoder.classes_

# -----------------------------
# Feature Scaling
# -----------------------------
features = ['ammonia', 'humidity', 'temperature', 'iaq', 'hour', 'daily_visitor']
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

# -----------------------------
# Create Sequences for CNN-LSTM
# -----------------------------
sequence_length = 60  # 60-min window
X, y = [], []
for i in range(sequence_length, len(scaled_features)):
    X.append(scaled_features[i-sequence_length:i])
    y.append(df['label'].iloc[i])
X, y = np.array(X), np.array(y)

# One-hot encode labels
y_cat = to_categorical(y)

# -----------------------------
# Build CNN-LSTM Model
# -----------------------------
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=64))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(X, y_cat, epochs=10, batch_size=64, validation_split=0.2)

# -----------------------------
# Evaluate and Predict
# -----------------------------
y_pred_prob = model.predict(X)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=label_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Sample Prediction Confidence
# -----------------------------
for i in range(5):
    print(f"Prediction: {label_names[y_pred[i]]}, Confidence: {np.max(y_pred_prob[i]) * 100:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
model.save("cnn_lstm_cleanliness_model.h5")
print("\n✅ Model saved as cnn_lstm_cleanliness_model.h5")
