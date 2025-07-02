
# ===================================================
# FILE A: Data Preprocessing + Feature Engineering + EDA for ML Toilet v2
# ===================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel("data_v1_15june.xlsx")

# -----------------------------
# Part 1: Preprocessing
# -----------------------------

# Drop rows with missing sensor values
df_clean = df.dropna(subset=["ammonia", "humidity", "temperature", "iaq"])

# Fill missing visitor values using forward/backward fill
df_clean["visitor"] = df_clean["visitor"].fillna(method='ffill').fillna(method='bfill')

# Extract time features
df_clean["day"] = df_clean["timestamp"].dt.day_name()
df_clean["hour"] = df_clean["timestamp"].dt.hour

# Convert visitor to daily only (max value per day)
daily_visitors = df_clean.groupby(df_clean["timestamp"].dt.date)["visitor"].transform("max")
df_clean["daily_visitor"] = daily_visitors

# Drop original visitor column
df_clean = df_clean.drop(columns=["visitor"])

# -----------------------------
# Part 2: Exploratory Data Analysis (EDA)
# -----------------------------

# Summary statistics
print("\n[INFO] Data Summary:")
print(df_clean.describe())

# Missing value check
print("\n[INFO] Missing Values:")
print(df_clean.isnull().sum())

# Distribution plots
plt.figure(figsize=(12, 6))
sns.histplot(df_clean["ammonia"], kde=True, bins=50)
plt.title("Ammonia Distribution")
plt.xlabel("Ammonia (ppm)")
plt.ylabel("Count")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean[["humidity", "temperature", "iaq"]])
plt.title("Boxplot: Humidity, Temperature, IAQ")
plt.grid(True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df_clean[["ammonia", "humidity", "temperature", "iaq", "daily_visitor"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Save preprocessed dataset
df_clean.to_csv("cleaned_data_for_ml.csv", index=False)
print("\n✅ Cleaned data saved as 'cleaned_data_for_ml.csv'")
