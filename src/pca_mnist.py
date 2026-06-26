# =====================================================
# IMPORTS
# =====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist

# =====================================================
# LOAD FASHION-MNIST DATASET
# =====================================================
try:
    # Attempt to load from user's CSV files
    print("Attempting to load data from CSV...")
    train_df = pd.read_csv('/content/fashion-mnist_train.csv').dropna().reset_index(drop=True)
    test_df = pd.read_csv('/content/fashion-mnist_test.csv').dropna().reset_index(drop=True)
    
    X_train_original = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_test_original = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values
    print("CSV loaded successfully.")

except FileNotFoundError:
    # Fallback to Keras built-in dataset if CSVs are missing
    print("CSV not found. Falling back to Keras built-in dataset...")
    (X_train_original, y_train), (X_test_original, y_test) = fashion_mnist.load_data()
    X_train_original = X_train_original.reshape(-1, 784)
    X_test_original = X_test_original.reshape(-1, 784)

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Visualize Samples
plt.figure(figsize=(12,5))
for i in range(10):
    idx = np.where(y_train == i)[0][0]
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train_original[idx].reshape(28,28), cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')
plt.suptitle("Fashion-MNIST Sample Classes")
plt.tight_layout()
plt.show()

# =====================================================
# NORMALIZATION
# =====================================================
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_original)
X_test = scaler.transform(X_test_original)

# =====================================================
# PCA (Dimensionality Reduction)
# =====================================================
print("\nRunning PCA...")
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original Dimension : {X_train.shape[1]}")
print(f"Reduced Dimension  : {X_train_pca.shape[1]}")

# 1. PCA Components Visualisation
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(pca.components_[i].reshape(28,28), cmap='gray')
    plt.title(f"PC {i+1}")
    plt.axis('off')
plt.suptitle("First 10 Principal Components")
plt.tight_layout()
plt.show()

# 2. Cumulative Variance Plot
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10,4))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained by PCA")
plt.grid(True)
plt.show()

# 3. 2D PCA Visualisation
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_train)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y_train, cmap='gnuplot2_r', s=5, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Fashion-MNIST PCA (2D Projection)")
plt.colorbar(scatter, label='Class')
plt.show()

# =====================================================
# NEURAL NETWORK SETUP & TRAINING (KERAS)
# =====================================================
n_inputs = X_train_pca.shape[1]
n_outputs = len(np.unique(y_train))

# Build Sequential Model
model = Sequential([
    Dense(15, activation='sigmoid', input_dim=n_inputs),
    Dense(16, activation='sigmoid'),
    Dense(n_outputs, activation='softmax')
])

# Compile Model
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
print("\nTraining Keras Neural Network...")
history = model.fit(X_train_pca, y_train, 
                    epochs=50, 
                    batch_size=32, # Replaced row-by-row with standard batch processing
                    validation_data=(X_test_pca, y_test), 
                    verbose=1)

# =====================================================
# TRAINING CURVES
# =====================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='orange')
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()

# =====================================================
# EVALUATION & METRICS
# =====================================================
test_loss, test_acc = model.evaluate(X_test_pca, y_test, verbose=0)
print(f"\n================================")
print(f"Test Accuracy = {test_acc * 100:.2f}%")
print(f"================================")

# Generate Predictions
y_pred_probs = model.predict(X_test_pca, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# 1. PCA Reconstruction Visualization
reconstructed_images = pca.inverse_transform(X_test_pca[:9])
plt.figure(figsize=(10, 6))
for i in range(9):
    # Original
    plt.subplot(3, 6, 2 * i + 1)
    plt.imshow(X_test_original[i].reshape(28, 28), cmap='gray')
    plt.title("Orig")
    plt.axis('off')
    
    # Reconstructed
    plt.subplot(3, 6, 2 * i + 2)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title("PCA")
    plt.axis('off')
plt.suptitle("Original vs PCA Reconstructed Images")
plt.tight_layout()
plt.show()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.show()

# 3. Classification Report
print("\nClassification Report\n")
print(classification_report(y_test, y_pred, target_names=class_names))

report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().iloc[:10]

plt.figure(figsize=(12,5))
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 5))
plt.title('Precision, Recall and F1-score by Class')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()
