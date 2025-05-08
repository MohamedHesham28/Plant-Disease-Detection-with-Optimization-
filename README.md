# Plant Disease Detection System

This project implements a deep learning-based plant disease detection system using PyTorch. It can identify various plant diseases from leaf images.

## Features

- CNN-based deep learning model for plant disease detection
- Support for 15 different plant diseases
- GPU acceleration support
- User-friendly GUI for image upload and prediction
- Model training and evaluation capabilities

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- tkinter
- numpy
- matplotlib
- tqdm

Install the requirements using:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the CNN architecture
- `train.py`: Training script
- `gui.py`: GUI application for predictions
- `utils.py`: Utility functions for data loading and preprocessing
- `requirements.txt`: Project dependencies

## Usage

### Training the Model

1. Ensure your dataset is organized in the following structure:
```
plant disease dataset/
    ├── train/
    │   ├── Pepper Bell Bacterial Spot/
    │   ├── Pepper Bell healthy/
    │   └── ...
    ├── val/
    │   ├── Pepper Bell Bacterial Spot/
    │   ├── Pepper Bell healthy/
    │   └── ...
    └── test/
        ├── Pepper Bell Bacterial Spot/
        ├── Pepper Bell healthy/
        └── ...
```

2. Run the training script:
```bash
python train.py
```

The script will automatically:
- Load and preprocess the data
- Train the model
- Save the best model to `plant_disease_model.pt`

### Making Predictions

1. Run the GUI application:
```bash
python gui.py
```

2. Click the "Upload Image" button to select a plant leaf image
3. The application will display:
   - The uploaded image
   - The predicted disease
   - The confidence score

## Model Architecture

The CNN model consists of:
- 4 convolutional layers with ReLU activation
- Max pooling layers
- Dropout for regularization
- 2 fully connected layers

## Hyperparameters

The model can be configured with the following hyperparameters:
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 10
- Dropout rate: 0.5

These can be adjusted in the `train.py` file.

## License

This project is licensed under the MIT License. 