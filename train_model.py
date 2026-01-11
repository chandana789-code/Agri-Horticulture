import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Example dummy model (replace with your real data/model)
model = Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Dummy training data
import numpy as np
X = np.random.rand(500, 100)
y = np.random.randint(0, 2, 500)

model.fit(X, y, epochs=5, batch_size=32)

# Save model
model.save("model.h5")

print("Model saved as model.h5")
