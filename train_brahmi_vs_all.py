import os
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 16

# -------------------------
# PATHS (your same paths)
# -------------------------
TAMIL_PATH = r"D:\NeoLipi_Dataset\tamil\tamil"
BRAHMI_PATH = r"D:\NeoLipi_Dataset\brahmi"
DEV_PATH = r"D:\NeoLipi_Dataset\devanagari\DevanagariHandwrittenCharacterDataset\Train"

# -------------------------
# GET ALL IMAGE PATHS
# -------------------------
def get_all_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.jpg','.png','.jpeg')):
                paths.append(os.path.join(root, file))
    return paths

print("Loading datasets...")
tamil_imgs = get_all_images(TAMIL_PATH)
brahmi_imgs = get_all_images(BRAHMI_PATH)
dev_imgs = get_all_images(DEV_PATH)

# -------------------------
# LABELS (IMPORTANT CHANGE)
# -------------------------
# Brahmi = 1, Others = 0
all_imgs = tamil_imgs + brahmi_imgs + dev_imgs

all_labels = (
    [0]*len(tamil_imgs) +      # Tamil → 0
    [1]*len(brahmi_imgs) +     # Brahmi → 1 ✅
    [0]*len(dev_imgs)          # Devanagari → 0
)

print("Total images:", len(all_imgs))

# -------------------------
# GENERATOR
# -------------------------
def data_generator(paths, labels, batch_size):
    while True:
        idx = np.random.permutation(len(paths))

        for i in range(0, len(paths), batch_size):
            batch_idx = idx[i:i+batch_size]

            X_batch = []
            y_batch = []

            for j in batch_idx:
                img = cv2.imread(paths[j], cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                img = img.reshape(IMG_SIZE, IMG_SIZE, 1)

                X_batch.append(img)
                y_batch.append(labels[j])

            yield np.array(X_batch), np.array(y_batch)

# -------------------------
# MODEL
# -------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# TRAIN
# -------------------------
steps = len(all_imgs) // BATCH_SIZE

print("Training Brahmi vs All...")
model.fit(
    data_generator(all_imgs, all_labels, BATCH_SIZE),
    steps_per_epoch=steps,
    epochs=5
)

# -------------------------
# SAVE
# -------------------------
model.save("brahmi_vs_all.h5")

print("✅ Brahmi model done")