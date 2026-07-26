import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# --------------------------------------------------------
# RANDOM SEED
# --------------------------------------------------------

tf.random.set_seed(42)
np.random.seed(42)

# --------------------------------------------------------
# DATASET PATH
# --------------------------------------------------------

DATASET_PATH = r"D:\NeoLipi_Dataset\tamil\train\augmented_images"

IMAGE_SIZE = (96, 96)

BATCH_SIZE = 64

EPOCHS = 30

# --------------------------------------------------------
# DATA AUGMENTATION
# --------------------------------------------------------

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    shear_range=0.10,
    fill_mode="nearest",
    validation_split=0.2
)

# --------------------------------------------------------
# LOAD DATASET
# --------------------------------------------------------

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes

print()

print("Classes :", NUM_CLASSES)

print("Images  :", train_generator.samples)

print()

print(train_generator.class_indices)
# --------------------------------------------------------
# LOAD PRETRAINED MOBILENETV2
# --------------------------------------------------------

print("\nLoading MobileNetV2...\n")

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(96,96,3)

)

# Freeze pretrained layers
base_model.trainable = False

print("MobileNetV2 Loaded Successfully!")

# --------------------------------------------------------
# BUILD OCR MODEL
# --------------------------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.4)(x)

x = Dense(
    512,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

predictions = Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# --------------------------------------------------------
# COMPILE MODEL
# --------------------------------------------------------

model.compile(

    optimizer=Adam(
        learning_rate=1e-4
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModel Compiled Successfully!\n")

model.summary()
# --------------------------------------------------------
# SAVE LABELS ONLY
# --------------------------------------------------------

class_names = list(train_generator.class_indices.keys())

with open("tamil_labels.txt", "w") as f:
    for label in class_names:
        f.write(label + "\n")

print("Tamil labels saved successfully!")

import sys
sys.exit()
# --------------------------------------------------------
# CALLBACKS
# --------------------------------------------------------

checkpoint = ModelCheckpoint(
    "tamil_ocr.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr
]

# --------------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------------

print("\n" + "="*60)
print("Starting OCR Training...")
print("="*60)

# history = model.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=EPOCHS,
#     callbacks=callbacks,
#     verbose=1
# )

# print("\nTraining Completed Successfully!")

# --------------------------------------------------------
# SAVE MODEL
# --------------------------------------------------------

model.save("tamil_ocr.keras")
model.save("tamil_ocr.h5")

print("\nOCR Model Saved Successfully!")

# --------------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------------

loss, accuracy = model.evaluate(
    train_generator,
    verbose=0
)

print("\nFinal Training Accuracy : {:.2f}%".format(accuracy*100))
print("Final Training Loss     : {:.4f}".format(loss))
# --------------------------------------------------------
# TRAINING ACCURACY GRAPH
# --------------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], linewidth=2)

plt.title("OCR Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("ocr_accuracy.png")
plt.show()

# --------------------------------------------------------
# TRAINING LOSS GRAPH
# --------------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], color="red", linewidth=2)

plt.title("OCR Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig("ocr_loss.png")
plt.show()

# --------------------------------------------------------
# SAVE CLASS LABELS
# --------------------------------------------------------

class_names = list(train_generator.class_indices.keys())

with open("tamil_labels.txt", "w") as f:
    for label in class_names:
        f.write(label + "\n")

print("\nLabels saved successfully!")
print("\n" + "="*60)
print("TAMIL OCR TRAINING COMPLETED")
print("="*60)

print(f"Dataset Path : {DATASET_PATH}")
print(f"Total Images : {train_generator.samples}")
print(f"Classes      : {NUM_CLASSES}")

print("\nGenerated Files:")
print("✓ brahmi_ocr.keras")
print("✓ brahmi_ocr.h5")
print("✓ brahmi_labels.txt")
print("✓ ocr_accuracy.png")
print("✓ ocr_loss.png")

print("\nFinal Accuracy : {:.2f}%".format(accuracy*100))
print("Final Loss     : {:.4f}".format(loss))

print("="*60)