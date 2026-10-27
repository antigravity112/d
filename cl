# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.datasets import fashion_mnist


# ============================================
# 2. CHECK TENSORFLOW VERSION
# ============================================

print("TensorFlow Version:", tf.__version__)


# ============================================
# 3. EXPLORE AVAILABLE LAYERS
# ============================================

print("\nAvailable common layers:")

print("Conv2D:", layers.Conv2D)
print("MaxPooling2D:", layers.MaxPooling2D)
print("Dense:", layers.Dense)
print("Dropout:", layers.Dropout)
print("Flatten:", layers.Flatten)
print("LSTM:", layers.LSTM)
print("Embedding:", layers.Embedding)
print("BatchNormalization:", layers.BatchNormalization)
print("GlobalAveragePooling2D:", layers.GlobalAveragePooling2D)


# ============================================
# 4. EXPLORE OPTIMIZERS
# ============================================

print("\nAvailable optimizers:")

print("Adam:", optimizers.Adam)
print("SGD:", optimizers.SGD)
print("RMSprop:", optimizers.RMSprop)
print("Adagrad:", optimizers.Adagrad)


# ============================================
# 5. EXPLORE LOSS FUNCTIONS
# ============================================

print("\nAvailable loss functions:")

print("Categorical Crossentropy:", losses.CategoricalCrossentropy)
print("Sparse Categorical Crossentropy:",
      losses.SparseCategoricalCrossentropy)
print("Binary Crossentropy:", losses.BinaryCrossentropy)
print("Mean Squared Error:", losses.MeanSquaredError)


# ============================================
# 6. EXPLORE ACTIVATION FUNCTIONS
# ============================================

print("\nAvailable activation functions:")

print("ReLU:", layers.ReLU)
print("Softmax:", layers.Softmax)
print("LeakyReLU:", layers.LeakyReLU)
print("ELU:", layers.ELU)


# ============================================
# 7. LOAD DATASET
# ============================================

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train[:1000].astype("float32") / 255.0
y_train = y_train[:1000]

x_test = x_test[:200].astype("float32") / 255.0
y_test = y_test[:200]

# Add channel dimension
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("\nTraining data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# ============================================
# 8. BUILD A SIMPLE CNN MODEL
# ============================================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])


# ============================================
# 9. DISPLAY MODEL ARCHITECTURE
# ============================================

model.summary()


# ============================================
# 10. COMPILE MODEL
# ============================================

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


# ============================================
# 11. TRAIN MODEL
# ============================================

history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2
)


# ============================================
# 12. EVALUATE MODEL
# ============================================

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================
# 13. PLOT TRAINING AND VALIDATION LOSS
# ============================================

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


# ============================================
# 14. MAKE A PREDICTION
# ============================================

prediction = model.predict(x_test[:1], verbose=0)

predicted_class = np.argmax(prediction[0])

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plt.imshow(x_test[0].squeeze(), cmap="gray")
plt.title("Predicted: " + class_names[predicted_class])
plt.axis("off")
plt.show()


# ============================================
# 15. EXPLORE MODEL METHODS
# ============================================

print("\nModel methods and attributes:")

print("model.fit:", model.fit)
print("model.predict:", model.predict)
print("model.evaluate:", model.evaluate)
print("model.summary:", model.summary)
