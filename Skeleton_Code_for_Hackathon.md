"""
Skeleton: Monkey Species Classifier
Goal: Train a model to classify monkey species from images using transfer learning.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetV2S
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Dataset Paths ===
TRAIN_DIR = "path_to_train_data"
TEST_DIR = "path_to_test_data"
CLASS_FILE = "class_names.json"

# === Load Dataset ===
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 32

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    label_mode="categorical",
    image_size=IMAGE_SIZE,
    shuffle=False
)

# === Save class names ===
class_names = train_dataset.class_names
with open(CLASS_FILE, "w") as f:
    json.dump(class_names, f)

# === Compute Class Weights (Hint: Use class_weight.compute_class_weight) ===
# train_labels = ...

# === Data Augmentation Pipeline ===
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# === Base Model + Custom Head ===
base_model = EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
base_model.trainable = True

for layer in base_model.layers[:150]:
    layer.trainable = False

inputs = Input(shape=(*IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=True)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Train the Model ===
# Optional: Add ReduceLROnPlateau or EarlyStopping if desired
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10,  # increase as needed
    # class_weight=class_weights_dict
)

# === Save the Trained Model ===
model.save("monkey_classifier.keras")

# === Confusion Matrix ===
# Hint: Loop through test_dataset to generate predictions and plot the matrix

# === Optional Prediction Function ===
def predict_monkey_species(image_path, model_path="monkey_classifier.keras", class_file="class_names.json"):
    with open(class_file, "r") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"🔍 Predicted monkey species: {predicted_class}")
    return predicted_class

# Example (Uncomment and use):
# predict_monkey_species("path/to/sample.jpg")
