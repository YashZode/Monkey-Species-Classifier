import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# Load your trained model
model_path = "fine_tuned_monkey_species.keras"  # Replace with your actual model file
model
# Define class labels (should match your training directory structure)
class_labels = [
    'Bald Uakari', 'Emperor Tamarin', 'Golden Monkey', 'Gray Langur',
    'Hamadryas Baboon', 'Mandril', 'Proboscis Monkey', 'Red Howler',
    'Vervet Monkey', 'White Faced Saki'
]

# Load and preprocess the test image
img_path = '/content/golden_monkey.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
predicted_label = class_labels[predicted_index]
confidence = float(predictions[0][predicted_index]) * 100

# Display result
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {predicted_label} ({confidence:.2f}%)')
plt.show()

print("Predicted Species:", predicted_label)
print("Confidence:", f"{confidence:.2f}%")