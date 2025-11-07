import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Paths to your dataset folders
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

# Target image size
IMG_SIZE = (48, 48)  # FER2013 standard

def load_dataset(base_path):
    """
    Loads images from folders. Each subfolder is an emotion class.
    Returns:
        X: NumPy array of shape (num_samples, 48, 48, 1)
        y: NumPy array of labels
    """
    X, y = [], []
    classes = sorted(os.listdir(base_path))  # ensure consistent label ordering
    print(f"Found classes: {classes}")

    for idx, emotion in enumerate(classes):
        folder = os.path.join(base_path, emotion)
        if not os.path.isdir(folder):
            continue
        print(f"Processing class '{emotion}' ({idx}) with {len(os.listdir(folder))} images...")
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            try:
                # Load image as grayscale and resize
                img = image.load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
                img_array = image.img_to_array(img) / 255.0  # normalize pixels
                X.append(img_array)
                y.append(idx)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Loaded {len(X)} images from {base_path}")
    return X, y

# Load train and test datasets
print("Loading training data...")
X_train, y_train = load_dataset(TRAIN_DIR)

print("\nLoading test data...")
X_test, y_test = load_dataset(TEST_DIR)

# Save as compressed NumPy files
np.savez_compressed("fer2013_train.npz", X=X_train, y=y_train)
np.savez_compressed("fer2013_test.npz", X=X_test, y=y_test)

print("\n✅ Dataset processing complete! Files saved:")
print("  - fer2013_train.npz")
print("  - fer2013_test.npz")
