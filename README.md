Fashion MNIST Classification using Principal Component Analysis (PCA) and Multi-Layer Perceptron (MLP)
Project Overview

This project implements an image classification system using the Fashion-MNIST dataset. The objective is to investigate whether dimensionality reduction using Principal Component Analysis (PCA) can reduce computational complexity while maintaining effective classification performance.

The original Fashion-MNIST images contain 784 pixel features (28 × 28 pixels). PCA is applied to reduce the feature space to 50 principal components, after which a custom Multi-Layer Perceptron (MLP) Neural Network is trained using feed-forward propagation and backpropagation.

The model achieves approximately 85% classification accuracy on unseen test data while reducing dimensionality by more than 93%.

Problem Statement

Image datasets often contain redundant and highly correlated features. Training neural networks on high-dimensional data increases computational cost and may introduce unnecessary complexity.

This project investigates whether PCA can effectively compress image data while preserving sufficient information for accurate classification.

Dataset Information

Dataset: Fashion-MNIST

Source:
https://github.com/zalandoresearch/fashion-mnist

Training Samples: 60,000

Testing Samples: 10,000

Image Size: 28 × 28 pixels

Number of Classes: 10

Classes:

T-shirt/Top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle Boot
Methodology
Dataset Loading
Data Visualization
Min-Max Normalization
Principal Component Analysis (PCA)
Neural Network Initialization
Model Training
Performance Evaluation
Result Interpretation

Workflow:

Fashion-MNIST Dataset
↓
Normalization
↓
PCA (784 → 50 Features)
↓
Custom MLP Neural Network
↓
Training
↓
Evaluation
↓
Prediction

Neural Network Architecture

Input Layer: 50 Neurons (PCA Features)

Hidden Layer 1: 15 Neurons

Hidden Layer 2: 16 Neurons

Output Layer: 10 Neurons

Activation Function: Sigmoid

Learning Rate: 0.01

Epochs: 50

Results
Metric	Value
Original Features	784
PCA Components	50
Dimensionality Reduction	93.62%
Test Accuracy	~85%
Epochs	50

Key Findings
PCA successfully reduced the dataset from 784 features to 50 principal components.
More than 93% of dimensionality was removed while retaining meaningful visual information.
The neural network achieved approximately 85% classification accuracy.
Trouser, Bag, and Ankle Boot classes achieved the highest classification performance.
Shirt, Pullover, and Coat classes were the most challenging due to visual similarity.
Applications
Fashion Product Classification
E-Commerce Product Tagging
Visual Search Systems
Recommendation Engines
Retail Inventory Automation
Edge AI Image Recognition
Future Scope
Convolutional Neural Networks (CNNs)
Deep Autoencoders
Hyperparameter Optimization
Transfer Learning
Real-Time Fashion Recognition Systems
Author

Mayur Shetty

M.Sc. Data Science and Artificial Intelligence
