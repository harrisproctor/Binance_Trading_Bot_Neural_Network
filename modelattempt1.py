import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import tensorflow as tf
from collections import Counter
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("btcusd_2024-01-01_to_2024-12-31.csv")

# Ensure time is in datetime format
df["time"] = pd.to_datetime(df["time"])

# Sort the data by time
df = df.sort_values("time")

df["date"] = df["time"].dt.date

# Group data by date
daily_groups = df.groupby("date")

# Extract features for each day
daily_data = []
for date, group in daily_groups:
    if len(group) == 1440:  # Only consider full days
        # Add additional features
        group["range"] = group["high"] - group["low"]  # Daily range
        group["change"] = group["close"] - group["open"]  # Change from open to close
        group["volatility"] = (group["high"] - group["low"]) / group["open"]  # Relative volatility

        # Extract features: close, range, change, volatility, and volume
        features = group[["close", "range", "change", "volatility", "volume"]].values
        daily_data.append(features)

# Convert to NumPy array
daily_data = np.array(daily_data)  # Shape: (num_days, 1440, num_features)
print(f"Shape of daily_data: {daily_data.shape}")

# Convert to NumPy array
daily_data = np.array(daily_data)
#print(daily_data)

# Extract daily close prices
daily_close_prices = df.groupby("date")["close"].last().values

# Calculate percentage change for the next day
labels = (daily_close_prices[1:] - daily_close_prices[:-1]) / daily_close_prices[:-1]

# Convert percentage changes to discrete classes
def label_mapper(change):
    if change > 0.05:        # >5% increase
        return 4             # Buy $50
    elif change > 0.04:
        return 3             # Buy $40
    elif change > 0.03:
        return 2             # Buy $30
    elif change > 0.02:
        return 1             # Buy $20
    elif change > 0.01:
        return 0             # Buy $10
    elif change < -0.05:     # <-5% decrease
        return 9             # Sell $50
    elif change < -0.04:
        return 8             # Sell $40
    elif change < -0.03:
        return 7             # Sell $30
    elif change < -0.02:
        return 6             # Sell $20
    elif change < -0.01:
        return 5             # Sell $10
    else:
        return 10            # Hold

# Apply the labeling function
discrete_labels = np.array([label_mapper(change) for change in labels])  # Shape: (num_days-1,)
print(discrete_labels)
# Align data with labels
X = daily_data[:-1]  # Shape: (num_days-1, 1440, num_features)
y = discrete_labels  # Shape: (num_days-1,)

# Function to visualize 3 random days
def visualize_random_days(daily_data, feature_names):
    # Choose three random days
    random_days = random.sample(range(len(daily_data)), 3)

    plt.figure(figsize=(16, 12))

    for i, day_idx in enumerate(random_days):
        day_data = daily_data[day_idx]  # Shape: (1440, num_features)

        # Create subplots for each feature
        for j, feature_name in enumerate(feature_names):
            plt.subplot(3, len(feature_names), i * len(feature_names) + j + 1)
            plt.plot(day_data[:, j], label=feature_name)
            plt.title(f"Day {day_idx + 1} - {feature_name}")
            plt.xlabel("Minute of the Day")
            plt.ylabel(feature_name)
            plt.grid()
            plt.tight_layout()

    plt.legend()
    plt.show()

# Feature names
feature_names = ["close", "range", "change", "volatility", "volume"]

# Call the function
visualize_random_days(daily_data, feature_names)




plt.figure(figsize=(8, 6))
sns.countplot(x=discrete_labels)
plt.title("Distribution of Labels")
plt.xlabel("Label (0=Buy $10, ..., 10=Hold)")
plt.ylabel("Frequency")
plt.show()

# Compute correlation matrix for one random day
random_day_idx = random.randint(0, len(daily_data) - 1)
day_data_df = pd.DataFrame(daily_data[random_day_idx], columns=["close", "range", "change", "volatility", "volume"])

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(day_data_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap for a Random Day")
plt.show()

y = discrete_labels # Simulated percentage changes

# Discretize continuous labels into 11 classes
bins = [-np.inf, -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, np.inf]
labels = [5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]  # 5-9: sell, 10: hold, 0-4: buy
y_discretized = pd.cut(discrete_labels, bins=bins, labels=labels).astype(int)  # Convert to discrete integers

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(daily_data, y_discretized, test_size=0.2, random_state=42)

# Build the LSTM model
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
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

# Make predictions on the test set
predictions = model.predict(X_test)  # Shape: (num_samples, 11)
predicted_classes = tf.argmax(predictions, axis=1).numpy()



# Convert predicted classes to human-readable actions
actions = []
for pred in predicted_classes:
    if pred <= 4:
        actions.append(f"Buy ${10 * (pred + 1)}")
    elif pred <= 9:
        actions.append(f"Sell ${10 * (pred - 4)}")
    else:
        actions.append("Hold")

# Display the first 10 predictions
print(f'len of pred classes: {len(predicted_classes)}')
print(len(actions))
for i, action in enumerate(actions[:10]):
    print(f"Day {i+1}: {action}")

# Count the occurrences of each class in the predictions
prediction_counts = Counter(predicted_classes)

# Display the distribution
total_predictions = len(predicted_classes)
for cls, count in prediction_counts.items():
    print(f"Class {cls}: {count} ({count / total_predictions:.2%})")

# Check the predicted probabilities for a sample
sample_probs = model.predict(X_test[:10])  # Predicted probabilities
print("Predicted Probabilities:\n", sample_probs)

# Check the predicted classes
sample_classes = tf.argmax(sample_probs, axis=1).numpy()
print("Predicted Classes:", sample_classes)

# Compare with true labels
print("True Labels:", y_test[:10])

model.save('my_1stmodel.keras')
