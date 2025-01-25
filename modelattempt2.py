import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("btcusd_2024-01-01_to_2024-12-31.csv")

# Ensure time is in datetime format
df["time"] = pd.to_datetime(df["time"])

# Sort the data by time
df = df.sort_values("time")
df["date"] = df["time"].dt.date

# Group data by date and extract features
daily_data = []
for date, group in df.groupby("date"):
    if len(group) == 1440:  # Ensure full days
        group["range"] = group["high"] - group["low"]
        group["change"] = group["close"] - group["open"]
        group["volatility"] = (group["high"] - group["low"]) / group["open"]
        features = group[["close", "range", "change", "volatility", "volume"]].values
        daily_data.append(features)

daily_data = np.array(daily_data)  # Shape: (num_days, 1440, 5)
print(f"Shape of daily_data: {daily_data.shape}")

# Extract daily close prices and calculate percentage change for labels
daily_close_prices = df.groupby("date")["close"].last().values
labels = (daily_close_prices[1:] - daily_close_prices[:-1]) / daily_close_prices[:-1]

# Map percentage changes to discrete classes
def label_mapper(change):
    bins = [-np.inf, -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, np.inf]
    labels = [5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
    return pd.cut([change], bins=bins, labels=labels).astype(int)[0]

discrete_labels = np.array([label_mapper(change) for change in labels])
print(f"Label distribution: {Counter(discrete_labels)}")

# Align X and y by slicing the last entry of y
X = daily_data[:-1]  # Remove the last day from daily_data
y = discrete_labels[:-1]  # Remove the last label to align with X

# Verify shapes
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Build LSTM model
model = Sequential([
    LSTM(128, input_shape=(1440, X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(11, activation="softmax")  # 11 classes
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training and validation metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

# Predict on test data
predictions = model.predict(X_test)
predicted_classes = tf.argmax(predictions, axis=1).numpy()

# Analyze predictions
prediction_counts = Counter(predicted_classes)
print(f"Prediction distribution: {prediction_counts}")

plt.figure(figsize=(8, 6))
sns.countplot(x=discrete_labels)
plt.title("Distribution of Labels")
plt.xlabel("Label (0=Buy $10, ..., 10=Hold)")
plt.ylabel("Frequency")
plt.show()


# Save the model
model.save('my_2ndmodel.keras')
