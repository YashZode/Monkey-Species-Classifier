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


### Download the dataset from the below URL
https://www.kaggle.com/datasets/utkarshsaxenadn/10-species-of-monkey-multiclass-classification

After unzipping the file 
– Delete “TFRecords Data” folder
– Delete “resnet50V2-v1.h5” file
– Keep the inner “Monkey Species Data” folder directly under the Python folder
• It will have “Prediction Data” and “Training Data” folders.


Remove the curropted files using the following code

    import os, glob
    files = glob.glob("Monkey Species Data/*/*/*")
    for file in files :
    f = open(file, "rb") # open to read binary file
    if not b"JFIF" in f.peek(10) :
    f.close()
    os.remove(file)
    else :
    f.close()

 
