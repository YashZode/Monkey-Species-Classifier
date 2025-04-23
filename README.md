# Monkey Species Classifier

This project uses a deep learning model to classify monkey species from images.

## 🧠 Model Overview
- Architecture: EfficientNetV2S
- Dataset: 10 species of monkeys
- Input size: 224x224
- Output: Species name + confidence score

## 📁 Project Structure
```
Monkey_Species_Data/
├── MonkeySpeciesData_1/
│   ├── Training Data/       # Labeled training images by species
│   └── Prediction Data/     # Unlabeled images to test predictions
├── saved_model.h5           # Trained Keras model
├── classify.py              # Code to predict species from new image
└── requirements.txt         # Required Python packages
```

## ✅ How to Use
### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Predict a Monkey Species
```bash
python classify.py --image sample_monkey.jpg
```

### Example Output
```
Predicted Species: Red Howler
Confidence: 96.57%
```

## 🐒 Species Supported
- Bald Uakari
- Emperor Tamarin
- Golden Monkey
- Gray Langur
- Hamadryas Baboon
- Mandril
- Proboscis Monkey
- Red Howler
- Vervet Monkey
- White Faced Saki

---

 