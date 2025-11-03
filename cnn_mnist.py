import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. Data Loading & Preparation ---
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# Normalization: Convert pixel values from [0, 255] to [0, 1]
X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0

# Reshaping/Channels: Reshape input to (samples, height, width, channels)
# MNIST images are 28x28 grayscale, so channel dimension is 1
X_train = np.expand_dims(X_train, -1) # (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, -1)   # (10000, 28, 28, 1)

# One-Hot Encoding: Convert integer labels (0-9) to one-hot vectors
y_train = to_categorical(y_train_raw, num_classes=10)
y_test = to_categorical(y_test_raw, num_classes=10)

print(f"Train data shape: {X_train.shape}, Train labels shape: {y_train.shape}")


# --- 2. Model Building (CNN) ---
model = Sequential([
    # First Conv -> MaxPool block
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    # Second Conv -> MaxPool block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Classification layers
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # 10 units for 10 classes (0-9) with softmax
])

# --- 3. Training & Evaluation ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining CNN Model...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Final Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy after 5 epochs: {accuracy*100:.2f}% (Goal > 95%)")

# --- 4. Visualization ---
# Take 5 random indices from the test set
random_indices = random.sample(range(X_test.shape[0]), 5)
sample_images = X_test[random_indices]
true_labels = y_test_raw[random_indices] # Use raw labels for display

# Run predictions
predictions = model.predict(sample_images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 4))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    # Reshape back for display (remove channel dim)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}",
              color=('green' if true_labels[i] == predicted_labels[i] else 'red'))
    plt.axis('off')

# plt.show() # Uncomment to display the plot
print("5 Random Test Images and their Predicted/True Labels generated.")