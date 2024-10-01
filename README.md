
# Handwritten Numbers (0-9) Image Classification

This project classifies handwritten digits using a convolutional neural network (CNN) built with PyTorch. The dataset is sourced from Kaggle and preprocessed for training, validation, and testing. The goal is to achieve high accuracy in digit recognition through deep learning techniques.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oafarih/Handwritten_Numbers_Image_Classification.git
   cd Handwritten_Numbers_Image_Classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset is downloaded from [Kaggle's Handwritten Digits 0-9](https://www.kaggle.com/datasets/olafkrastovski/handwritten-digits-0-9). It contains images of handwritten digits (0-9), divided into training, validation, and testing sets.

The dataset is processed using the `download_data.ipynb` file:
- Splits data into 80% training, 10% validation, and 10% testing sets.
- Images are resized and stored in grayscale.

## Model Architecture

The model is a CNN built with PyTorch:
- **Conv2d layers**: Extract spatial features.
- **ReLU activation**: Introduce non-linearity.
- **MaxPool2d layers**: Downsample the feature maps.
- **Fully connected layers**: Output class probabilities (0-9).

```python
class DigitsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )
```

## Training and Evaluation

The model is trained using the Adam optimizer and cross-entropy loss. The training process involves 10 epochs with batch sizes of 64.

Key metrics:
- Training accuracy and loss
- Validation accuracy and loss
- Test accuracy

Training and validation are handled by the `train_model()` function in `model_handwritten_digits.ipynb`. Test accuracy is evaluated using the `test_model()` function.

## Usage

1. Download the dataset by running the `download_data.ipynb` notebook.
2. Train the model using the `model_handwritten_digits.ipynb` notebook.

## Results

After 10 epochs, the model achieved a test accuracy of **97.72%**.

## License

This project is licensed under the MIT License.
