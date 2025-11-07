import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tqdm import tqdm  # pip install tqdm

# Load test data
data = np.load("fer2013_test.npz")
X_test, y_test = data["X"], data["y"]

# Take a smaller subset for faster evaluation
subset_size = 1000
X_test = X_test[:subset_size]
y_test = y_test[:subset_size]

# Convert grayscale (48x48x1) to RGB (96x96x3)
X_test = tf.image.resize(X_test, (96, 96))  # Resize
X_test = np.repeat(X_test, 3, axis=-1)      # Duplicate grayscale into 3 channels

# Load trained model
model = load_model("emotion_transfer_final.h5")

# Evaluate model in batches
batch_size = 32
y_pred = []
for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
    batch = X_test[i:i+batch_size]
    preds = np.argmax(model.predict(batch, verbose=0), axis=1)
    y_pred.extend(preds)
y_pred = np.array(y_pred)

# Convert y_test to labels if it's one-hot
if y_test.ndim > 1:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test

# Classification report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
