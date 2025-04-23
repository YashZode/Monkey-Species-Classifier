# monkey_species_trainer.py

import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense,
                                     Dropout)
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing import image_dataset_from_directory

# === Paths ===
TRAIN_DIR = "/content/Monkey_Species_Data/MonkeySpeciesData_1/Prediction Data"
TEST_DIR = "/content/Monkey_Species_Data/MonkeySpeciesData_1/Prediction Data"
CLASS_FILE = "/content/Monkey_Species_Data/class_names.json"

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

class_names = train_dataset.class_names
print("\n✅ Class names:", class_names)

with open(CLASS_FILE, "w") as f:
    json.dump(class_names, f)

# === Compute Class Weights ===
train_labels = [np.argmax(label.numpy()) for _, label in train_dataset.unbatch()]
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(weights))

# === Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# === EfficientNetV2S Model ===
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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print("\n🧪 Training EfficientNetV2S model...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=[reduce_lr]
)

# === Save Model ===
model.save("fine_tuned_monkey_species.keras")
print("\n✅ Model saved as fine_tuned_monkey_species.keras")

# === Confusion Matrix ===
y_true, y_pred = [], []
for images, labels in test_dataset:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# === Prediction Function ===

def predict_monkey_species(image_path, model_path="fine_tuned_monkey_species.keras", class_file="/content/Monkey_Species_Data/class_names.json"):
    with open(class_file, "r") as f:
        class_names = json.load(f)

    model = load_model(model_path)
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"\n🔍 Predicted monkey species: {predicted_class}")
    return predicted_class
    # Example:
predict_monkey_species("/content/sample_monkey.jpg")