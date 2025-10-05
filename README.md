# CyberSecurity_Assignment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import BinaryCrossentropy

# Generate synthetic IoT network traffic features
np.random.seed(42)
n_samples = 2000
# Features: duration, src_bytes, dst_bytes, packets (random floats/ints)
duration = np.random.uniform(0, 10, n_samples)
src_bytes = np.random.randint(0, 10000, n_samples)
dst_bytes = np.random.randint(0, 15000, n_samples)
packets = np.random.randint(1, 50, n_samples)

# Binary labels (0 = benign, 1 = attack) randomly assigned
labels = np.random.choice([0,1], size=n_samples, p=[0.7, 0.3])

# Create DataFrame
data = pd.DataFrame({
    "duration": duration,
    "src_bytes": src_bytes,
    "dst_bytes": dst_bytes,
    "packets": packets,
    "label": labels
})

# Save to CSV
data.to_csv("iot_data_synthetic.csv", index=False)
print("Synthetic IoT dataset saved as iot_data_synthetic.csv")

# Load dataset
data = pd.read_csv("iot_data_synthetic.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Build deep learning model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Focal Loss implementation
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        bce = BinaryCrossentropy()(y_true, y_pred)
        pt = tf.exp(-bce)
        return alpha * (1 - pt) ** gamma * bce
    return loss

model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=64)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc}")


