# MNIST CNN Classification

A PyTorch implementation of a Convolutional Neural Network for MNIST digit classification with Weights & Biases integration for experiment tracking.

<div align="center">
    <img src="assets/sample_predictions.png" alt="Sample predictions visualization" width=80%>
    <p><em>Example predictions from the trained model on MNIST test set</em></p>
</div>

## Overview

This project implements a CNN architecture for the MNIST handwritten digit classification task. The implementation focuses on exploring various architectural choices and techniques including:

- Using padding to maintain feature map dimensions
- Batch normalization vs. layer normalization before and after activation functions
- Strided convolutions vs. max/average pooling for downsampling
- Learning rate scheduling and weight decay regularization
- Data augmentation with affine transformations
- Experiment tracking with Weights & Biases

Through systematic experimentation, the combination of **strided convolutions with batch normalization before ReLU activations proved most effective for this task, achieving >99% accuracy on the MNIST test set.**

### Experiment Tracking
All training runs are logged to Weights & Biases, capturing:
- Training and test accuracy/loss curves
- Real-time prediction visualizations
- Misclassification analysis
- Hyperparameter configurations


[View detailed experiment results on Weights & Biases](wandb.ai/noahp/MNIST%20CNN%20CS499%20A3)

## Project Components

### Data Processing
- Standard MNIST dataset loading via PyTorch's datasets module
- Data normalization
- Data augmentation with random affine transformations (rotation, scaling, translation)

### Model Architecture
`SimpleCNN`: A convolutional neural network with:

**Block 1:**
- Conv2d(1→36, 5×5) → BatchNorm2d → ReLU
- Conv2d(36→36, 5×5) → BatchNorm2d → ReLU
- Strided Conv2d(36→36, 2×2, stride=2) for downsampling

**Block 2:**
- Conv2d(36→64, 3×3) → BatchNorm2d → ReLU
- Conv2d(64→128, 3×3) → BatchNorm2d → ReLU
- Strided Conv2d(128→128, 2×2, stride=2) for downsampling

**Block 3:**
- Conv2d(128→256, 1×1) → BatchNorm2d → ReLU
- Conv2d(256→10, 1×1)
- AdaptiveAvgPool2d to reduce spatial dimensions to 1×1

### Training & Evaluation
- Cross-entropy loss for classification
- Adam optimizer with exponential learning rate scheduling
- Visualization of model predictions and misclassifications with softmax output
- Integration with Weights & Biases (wandb) for experiment tracking

### Visualization

The code includes several visualization tools:
- Data augmentation preview
- Real-time prediction visualization during training (shows images and corresponding softmax outputs)
- Misclassification analysis at the end of training (shows images and corresponding softmax outputs)

The prediction and misclassification visualizations are automatically logged to your Weights & Biases project dashboard.

## Results & Analysis

The model achieves > 99% accuracy on the MNIST test set with the following key findings:

1. **Strided Convolutions vs. Max Pooling**: Strided convolutions provided better performance than max pooling for downsampling operations.

2. **Normalization**: Batch normalization consistently outperformed layer normalization in this architecture.

3. **Data Augmentation**: Adding random affine transformations improved model generalization and reduced overfitting.

## Usage

1. Clone the repository
2. Install the required dependencies: `pip install -r ../requirements.txt`
3. Make sure you have a Weights & Biases account and are logged in (`wandb login`)
4. Run the training script: `python MNIST_CNN.py`

### Configuration

Modify the `config` dictionary to adjust hyperparameters:

```python
config = {
    "bs": 2048,       # batch size
    "lr": 0.003,      # learning rate
    "l2reg": 0.00005, # weight decay
    "lr_decay": 0.99, # exponential learning rate decay
    "aug": True,      # enable data augmentation
    "max_epoch": 20   # number of training epochs
}
```

Modify `SimpleCNN` to adjust the model architecture. To modify the architecture, you can:

- Uncomment the layer normalization lines and comment out batch normalization
- Uncomment max pooling lines and comment out strided convolution
- Adjust the filter sizes, strides, and channel counts