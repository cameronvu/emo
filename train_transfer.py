import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import cv2

# Load processed data
train_data = np.load("fer2013_train.npz")
test_data = np.load("fer2013_test.npz")

X_train, y_train = train_data['X'], train_data['y']
X_test, y_test = test_data['X'], test_data['y']

# Resize and convert grayscale -> RGB (3 channels)
def preprocess_images(X):
    X_resized = []
    for img in X:
        # original shape: (48,48,1)
        img = img.reshape(48,48)  # drop channel
        img = cv2.resize(img, (96,96))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 3 channels
        X_resized.append(img_rgb)
    return np.array(X_resized) / 255.0  # normalize 0-1

X_train = preprocess_images(X_train)
X_test = preprocess_images(X_test)

# One-hot encode labels
num_classes = 7
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers (optional: you can unfreeze later for fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('emotion_transfer.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)

# Train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[checkpoint, earlystop]
)

# Save final model
model.save("emotion_transfer_final.h5")
print("Training complete! Model saved as emotion_transfer_final.h5")
