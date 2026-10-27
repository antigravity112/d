
# ============================================
# CNN FOR FASHION-MNIST WITH ALL IMPORTANT PLOTS
# ============================================

# 1. IMPORT LIBRARIES
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 2. CHECK TENSORFLOW VERSION
print("TensorFlow Version:", tf.__version__)

# 3. EXPLORE LAYERS, OPTIMIZERS, LOSSES AND ACTIVATIONS
print("\nLayers:", layers.Conv2D, layers.Dense, layers.Dropout)
print("Optimizers:", optimizers.Adam, optimizers.SGD, optimizers.RMSprop)
print("Losses:", losses.SparseCategoricalCrossentropy,
      losses.BinaryCrossentropy, losses.MeanSquaredError)
print("Activations:", layers.ReLU, layers.Softmax, layers.LeakyReLU)

# 4. LOAD DATASET
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train[:1000].astype("float32") / 255.0
y_train = y_train[:1000]

x_test = x_test[:200].astype("float32") / 255.0
y_test = y_test[:200]

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 5. BUILD CNN MODEL
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

# 6. DISPLAY MODEL
model.summary()

# 7. COMPILE
model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# 8. TRAIN
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=32,
    validation_split=0.2
)

# 9. EVALUATE
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# 10. PLOT TRAINING AND VALIDATION LOSS
plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# 11. PLOT TRAINING AND VALIDATION ACCURACY
plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid()
plt.show()

# 12. PREDICT ALL TEST IMAGES
probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(probabilities, axis=1)

# 13. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred, labels=range(10))

plt.figure(figsize=(9, 8))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Fashion-MNIST Confusion Matrix")
plt.tight_layout()
plt.show()

# 14. DISPLAY SAMPLE PREDICTIONS
plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")

    actual = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]

    color = "green" if y_test[i] == y_pred[i] else "red"

    plt.title(
        f"Pred: {predicted}\nActual: {actual}",
        color=color,
        fontsize=9
    )
    plt.axis("off")

plt.suptitle("CNN Predictions: Green = Correct, Red = Incorrect")
plt.tight_layout()
plt.show()

# 15. VISUALIZE FIRST CONVOLUTION LAYER FEATURE MAPS
feature_model = models.Model(
    inputs=model.inputs,
    outputs=model.layers[0].output
)

feature_maps = feature_model.predict(x_test[:1], verbose=0)

plt.figure(figsize=(10, 5))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
    plt.axis("off")
    plt.title(f"Filter {i + 1}")

plt.suptitle("Feature Maps from First Conv2D Layer")
plt.tight_layout()
plt.show()

# 16. EXPLORE MODEL METHODS
print("\nModel methods:")
print("model.fit:", model.fit)
print("model.predict:", model.predict)
print("model.evaluate:", model.evaluate)
print("model.summary:", model.summary)

# ============================================
# 17. PREDICT THE WORD IN A SENTENCE
# ============================================

index = 0

image = x_test[index:index + 1]

prediction = model.predict(image, verbose=0)

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
print(f"Prediction confidence: {confidence:.2f}%")

# Display the image with the sentence
plt.figure(figsize=(5, 5))
plt.imshow(x_test[index].squeeze(), cmap="gray")

plt.title(
    f"The image shows a {class_names[predicted_class]}.\n"
    f"Actual: {class_names[actual_class]} | "
    f"Confidence: {confidence:.2f}%"
)

plt.axis("off")
plt.show()
