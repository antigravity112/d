
# ============================================
# CNN-LSTM FOR FASHION-MNIST CLASSIFICATION
# ============================================

# 1. IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ============================================
# 2. CHECK TENSORFLOW VERSION
# ============================================

print("TensorFlow Version:", tf.__version__)

# ============================================
# 3. EXPLORE AVAILABLE LAYERS
# ============================================

print("\nAvailable layers:")
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

# Add channel dimension for CNN
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("\nTraining data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# ============================================
# 8. BUILD CNN-LSTM MODEL (CORRECTED)
# ============================================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Convert image into 28 time steps
    # Each step has shape (28, 1)
    layers.Reshape((28, 28, 1)),

    # Add a feature dimension to each row
    layers.TimeDistributed(
        layers.Reshape((28, 1, 1))
    ),

    # CNN extracts features from each row
    layers.TimeDistributed(
        layers.Conv2D(16, (3, 1), activation="relu")
    ),

    layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    ),

    # LSTM learns sequential patterns
    layers.LSTM(64),

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),

    # Classify into 10 clothing categories
    layers.Dense(10, activation="softmax")
])

model.summary()

# Compile
model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)


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

test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ============================================
# 13. PLOT TRAINING AND VALIDATION LOSS
# ============================================

plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN-LSTM Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# ============================================
# 14. PLOT TRAINING AND VALIDATION ACCURACY
# ============================================

plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN-LSTM Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# ============================================
# 15. PREDICT TEST IMAGES
# ============================================

probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

# ============================================
# 16. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(y_test, y_pred, labels=range(10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("CNN-LSTM Confusion Matrix")
plt.tight_layout()
plt.show()

# ============================================
# 17. DISPLAY SAMPLE PREDICTIONS
# ============================================

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(x_test[i].squeeze(), cmap="gray")

    actual = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]

    color = "green" if y_test[i] == y_pred[i] else "red"

    plt.title(
        f"Predicted: {predicted}\nActual: {actual}",
        color=color,
        fontsize=9
    )

    plt.axis("off")

plt.suptitle("CNN-LSTM Predictions (Green = Correct)")
plt.tight_layout()
plt.show()



# ============================================
# 18. VISUALIZE CNN FEATURE MAPS (FINAL FIX)
# ============================================

# Select the CNN output layer
feature_model = models.Model(
    inputs=model.inputs,
    outputs=model.layers[3].output
)

feature_maps = feature_model.predict(x_test[:1], verbose=0)

print("Feature map shape:", feature_maps.shape)

# Shape is (1, 28, 16)
plt.figure(figsize=(12, 6))

for i in range(min(16, feature_maps.shape[-1])):
    plt.subplot(4, 4, i + 1)

    # Display one filter across all 28 time steps
    plt.plot(feature_maps[0, :, i])

    plt.title(f"Filter {i + 1}")
    plt.xlabel("Image Row")
    plt.ylabel("Activation")
    plt.grid(True)

plt.suptitle("CNN Filter Activations Across Image Rows")
plt.tight_layout()
plt.show()


# ============================================
# 19. PREDICT THE CLOTHING WORD IN A SENTENCE
# ============================================

index = 0

prediction = model.predict(x_test[index:index + 1], verbose=0)

predicted_class = np.argmax(prediction[0])
confidence = prediction[0][predicted_class] * 100
actual_class = y_test[index]

print("\nPREDICTION OUTPUT")
print("------------------------------")

print(
    f"The model predicts that the image shows a "
    f"{class_names[predicted_class]}."
)

print(f"Actual item: {class_names[actual_class]}")
print(f"Confidence: {confidence:.2f}%")

plt.figure(figsize=(5, 5))
plt.imshow(x_test[index].squeeze(), cmap="gray")

plt.title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Actual: {class_names[actual_class]}\n"
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")
plt.show()

# ============================================
# 20. EXPLORE MODEL METHODS
# ============================================

print("\nModel methods and attributes:")
print("model.fit:", model.fit)
print("model.predict:", model.predict)
print("model.evaluate:", model.evaluate)
print("model.summary:", model.summary)
