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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow.keras.backend as K

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred)
        return K.mean(K.sum(loss, axis=-1))
    return loss


# Load data
df = pd.read_csv("btcusd_2022-01-01_to_2024-12-31.csv")

# Ensure time is in datetime format
df["time"] = pd.to_datetime(df["time"])

# Sort the data by time
df = df.sort_values("time")

# Create new features
df["range"] = df["high"] - df["low"]
df["change"] = df["close"] - df["open"]
df["volatility"] = (df["high"] - df["low"]) / df["open"]

# Extract features
features = df[["close", "range", "change", "volatility", "volume"]].values
print(f"Shape of features: {features.shape}")  # Shape: (num_samples, 5)

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Define window sizes
timesteps_per_hour = 60  # Minutes in an hour
num_timesteps = len(features) // timesteps_per_hour

X = []
y = []

# Loop to create input (1-hour blocks) and labels (next hour's price change)
for i in range(num_timesteps - 1):  # Subtract 1 to ensure space for the next hour
    # Create a 1-hour block of features
    hour_block = features_normalized[i * timesteps_per_hour: (i + 1) * timesteps_per_hour]
    X.append(hour_block)

    # Calculate the percentage change in close price for the next hour
    current_close = df.iloc[(i + 1) * timesteps_per_hour - 1]["close"]
    next_close = df.iloc[(i + 2) * timesteps_per_hour - 1]["close"]
    percent_change = (next_close - current_close) / current_close

    # Map the change to discrete labels
    y.append(0 if percent_change < 0 else 1)

X = np.array(X)  # Shape: (num_samples, 60, 5)
y = np.array(y)  # Shape: (num_samples,)

# Verify shapes
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check label distribution
print(f"Label distribution: {Counter(y)}")

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"Computed Class Weights: {class_weights}")

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Convert labels to one-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=2)
y_test_encoded = to_categorical(y_test, num_classes=2)

# Build LSTM model
model = Sequential([
    LSTM(128, input_shape=(1440, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax")
])
modelo = Sequential([
    LSTM(128, input_shape=(1440, X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax")  # 11 classes
])
# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Train the model with class weights
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_test, y_test_encoded),
    epochs=25,
    batch_size=32,
    #class_weight=class_weights,  # Apply class weights
    #callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")

# Plot training and validation metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy with Class Weights")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss with Class Weights")
plt.show()

# Predict on test data
predictions = model.predict(X_test)
predicted_classes = tf.argmax(predictions, axis=1).numpy()

# Analyze predictions
prediction_counts = Counter(predicted_classes)
actual_counts = Counter(y_test)

# Calculate proportions
total_predictions = sum(prediction_counts.values())
total_actuals = sum(actual_counts.values())

predicted_proportions = {label: count / total_predictions for label, count in prediction_counts.items()}
actual_proportions = {label: count / total_actuals for label, count in actual_counts.items()}

# Print distributions
print(f"Prediction distribution (counts): {prediction_counts}")
print(f"Actual distribution (counts): {actual_counts}")
print(f"Prediction distribution (proportions): {predicted_proportions}")
print(f"Actual distribution (proportions): {actual_proportions}")

# Visualize distributions
labels = sorted(set(y_test))  # Ensure consistent order of labels
predicted_values = [predicted_proportions.get(label, 0) for label in labels]
actual_values = [actual_proportions.get(label, 0) for label in labels]

x = np.arange(len(labels))  # Label positions

plt.figure(figsize=(10, 6))
bar_width = 0.35

# Plot predicted proportions
plt.bar(x - bar_width / 2, predicted_values, bar_width, label="Predicted Proportions", color="blue")

# Plot actual proportions
plt.bar(x + bar_width / 2, actual_values, bar_width, label="Actual Proportions", color="orange")

# Add labels and legend
plt.xticks(x, labels)
plt.title("Comparison of Predicted and Actual Label Proportions")
plt.xlabel("Labels")
plt.ylabel("Proportion")
plt.legend()

# Show plot
plt.show()

y_pred = tf.argmax(model.predict(X_test), axis=1).numpy()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()


# Save the model
model.save('my_5thmodel_hourly.keras')
