import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Create model
# -----------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Dummy training data
# -----------------------------
X = np.random.rand(500, 3)
y = np.random.randint(0, 2, 500)

model.fit(X, y, epochs=10, batch_size=32)

# -----------------------------
# Save model
# -----------------------------
model.save("model.h5")

print("Model trained and saved as model.h5")
