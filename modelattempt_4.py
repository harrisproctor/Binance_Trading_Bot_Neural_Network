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
df["date"] = df["time"].dt.date

# Group data by date and extract features
daily_data = []
valid_dates = []  # Keep track of valid dates
for date, group in df.groupby("date"):
    if len(group) == 1440:  # Ensure full days
        group["range"] = group["high"] - group["low"]
        group["change"] = group["close"] - group["open"]
        group["volatility"] = (group["high"] - group["low"]) / group["open"]
        features = group[["close", "range", "change", "volatility", "volume"]].values
        daily_data.append(features)
        valid_dates.append(date)  # Add the date to valid_dates

daily_data = np.array(daily_data)  # Shape: (num_days, 1440, 5)
print(f"Shape of daily_data: {daily_data.shape}")

# Flatten daily data for normalization
scaler = StandardScaler()
daily_data_flat = daily_data.reshape(-1, daily_data.shape[-1])  # Shape: (num_samples * timesteps, num_features)
daily_data_normalized = scaler.fit_transform(daily_data_flat)
daily_data = daily_data_normalized.reshape(daily_data.shape)  # Reshape back

# Filter daily_close_prices to include only valid_dates
daily_close_prices = df.groupby("date")["close"].last()
daily_close_prices = daily_close_prices.loc[valid_dates].values  # Use only valid dates

# Calculate percentage change for labels
labels = (daily_close_prices[1:] - daily_close_prices[:-1]) / daily_close_prices[:-1]

# Map percentage changes to discrete classes
def label_mapper(change):
    bins = [-np.inf, 0, np.inf]
    labels = [0, 1]
    return pd.cut([change], bins=bins, labels=labels).astype(int)[0]

discrete_labels = np.array([label_mapper(change) for change in labels])
print(f"Label distribution: {Counter(discrete_labels)}")

# Align X and y by slicing the last entry of y
X = daily_data[:-1]  # Remove the last day from daily_data
y = discrete_labels  # Already aligned with valid days

# Verify shapes
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
modelold = Sequential([
    LSTM(128, input_shape=(1440, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax")
])
model = Sequential([
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
    epochs=50,
    batch_size=32,
    class_weight=class_weights,  # Apply class weights
    callbacks=[early_stopping]
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
model.save('my_4thmodel.keras')
