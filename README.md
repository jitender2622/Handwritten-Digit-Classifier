# Learn PyTorch: MNIST Digit Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jitender2622/Learn-pytorch-mnist/blob/main/Learn_pytorch_mnist.ipynb)

This project demonstrates a fundamental implementation of a Neural Network using **PyTorch** to classify handwritten digits from the famous **MNIST dataset**. It serves as a clean, beginner-friendly template for understanding the PyTorch Deep Learning pipeline.

## 📄 Project Overview

The goal is to train a machine learning model to look at a 28x28 pixel image of a handwritten number (0-9) and correctly identify which number it is.



The project covers the complete workflow:
1.  **Data Ingestion:** Downloading and transforming the MNIST data.
2.  **Model Design:** Building a Feed-Forward Neural Network (Multi-Layer Perceptron).
3.  **Training:** Implementing the backpropagation optimization loop.
4.  **Inference:** Visualizing predictions on unseen test data.

## 🧠 Model Architecture

The model used is a simple fully connected network (MLP) defined in `MNISTClassifier`:

* **Input Layer:** 784 nodes (Flattened 28x28 images).
* **Hidden Layer:** 128 nodes with **ReLU** activation.
* **Output Layer:** 10 nodes (representing probabilities for digits 0-9).



## 🛠 Tech Stack

* **Python 3.x**
* **PyTorch:** Core deep learning framework.
* **Torchvision:** For loading datasets and image transforms.
* **Matplotlib:** For visualizing the test results.
* **Google Colab:** Recommended environment (supports free GPU).

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge at the top of this README. The notebook is pre-configured to:
* Auto-detect if a GPU is available.
* Download the dataset automatically.
* Visualize the final predictions.

### Option 2: Local Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/jitender2622/Learn-pytorch-mnist.git](https://github.com/jitender2622/Learn-pytorch-mnist.git)
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision matplotlib
    ```
3.  Run the notebook using Jupyter:
    ```bash
    jupyter notebook Learn_pytorch_mnist.ipynb
    ```

## 📊 Performance & Results

The training loop runs for **5 epochs** by default.

* **Optimizer:** Adam (Learning Rate: 0.001)
* **Loss Function:** CrossEntropyLoss
* **Expected Accuracy:** You should observe a test accuracy of approximately **97-98%** after 5 epochs.

### Visual Test
The notebook concludes with a visualization block that displays 5 random images from the test set alongside the model's prediction and the actual true label.

## 🤝 Contributing

Feel free to fork this project and experiment with adding Convolutional layers (CNNs) to improve accuracy!
