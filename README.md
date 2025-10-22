# Food Classification with PyTorch

A deep learning project for food image classification using PyTorch and Computer Vision techniques, trained on 40 randomly selected classes from the Food-101 dataset.

## 📋 Description

This project builds a deep learning model to classify 40 different types of food from images. The model is trained using PyTorch with modern Computer Vision techniques on a subset of the Food-101 dataset.

## 🚀 Features

- Classifies 40 randomly selected food categories
- Built with PyTorch framework
- Pre-trained model available (Food_101_Pytorch_CV_full.pth)
- Supports inference on new images
- Efficient training on reduced dataset

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/jeff454/ComputerVision.git
cd ComputerVision

# Install dependencies
pip install torch torchvision pillow numpy matplotlib
```

## 💻 Usage
```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('Food_101_Pytorch_CV_full.pth', map_location=device)
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('your_food_image.jpg')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)
    
print(f"Predicted class: {prediction.item()}")
```

## 📊 Dataset

- **Source**: Food-101 Dataset
- **Classes Used**: 40 randomly selected categories from 101 total classes
- **Images per class**: 
  - Training: 750 images
  - Testing: 250 images
- **Total images**: ~40,000 images

### Food Categories
- Apple pie
- Pizza
- Hamburger
- Sushi
- ...

## 🛠️ Tech Stack

- **Python** 3.8+
- **PyTorch** - Deep learning framework
- **Torchvision** - Image transformations and models
- **PIL/Pillow** - Image processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization

## 🏗️ Model Architecture

- Base Model: ResNet-50 / VGG16 / Custom CNN
- Input Size: 224x224x3
- Output: 40 classes
- Transfer Learning: [Yes/No]

## 📈 Training Details

- **Optimizer**: Adam 
- **Learning Rate**: [0.0003]
- **Batch Size**: [32]
- **Epochs**: [20]
- **Loss Function**: [CrossEntropyLoss]
- **Data Augmentation**:[RandomHorizontalFlip]

## 📊 Results

- **Final Training Accuracy**: 99.71%
- **Best Test Accuracy**: 87.81%
- **Total Epochs**: 17/20 (Early stopping)
- **Best Model Saved at**: Epoch 12



## 📁 Project Structure
```
ComputerVision/
├── Food_101_Pytorch_CV_full.pth   # Trained model weights
├── .gitignore                      # Git ignore file
├── README.md                       # Project documentation
├── train.py                        # Training script (if applicable)
├── inference.py                    # Inference script (if applicable)
└── requirements.txt                # Python dependencies
```

## 🚦 Quick Start

1. Clone the repository
2. Install dependencies
3. Download the Food-101 dataset (if needed)
4. Run inference with the pre-trained model
5. Or train your own model with custom parameters

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

- **jeff454**
- GitHub: [@jeff454](https://github.com/jeff454)

## 🙏 Acknowledgments

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) by ETH Zurich
- [PyTorch Documentation](https://pytorch.org/docs/)
- PyTorch Community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project helpful, please give it a star!
