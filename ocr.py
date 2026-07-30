"""
===========================================================
NeoLipi OCR Module
Supports:
    • Brahmi
    • Tamil
    • Devanagari
===========================================================
"""

import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

# ==========================================================
# LOAD MODELS
# ==========================================================

print("\nLoading OCR Models...\n")

BRAHMI_MODEL = tf.keras.models.load_model(
    MODEL_DIR / "best_ocr_model.keras"
)

TAMIL_MODEL = tf.keras.models.load_model(
    MODEL_DIR / "tamil_model.keras"
)

DEVANAGARI_MODEL = tf.keras.models.load_model(
    MODEL_DIR / "devanagari_model.keras"
)

print("OCR Models Loaded Successfully.")

# ==========================================================
# LOAD CLASS FILES
# ==========================================================

# ==========================================================
# LOAD CLASS FILES
# ==========================================================

def load_class_mapping(file_path):
    """
    Converts any supported JSON format into
    index -> label mapping.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format 1
    # ["a","b","c"]

    if isinstance(data, list):
        return data

    # Format 2
    # {"Brahmi":0,"Tamil":1}

    # Format 3
    # {"ka":10,"kha":11}

    mapping = {}

    for key, value in data.items():
        mapping[int(value)] = key

    return mapping


BRAHMI_CLASSES = load_class_mapping(
    MODEL_DIR / "brahmi_classes.json"
)

TAMIL_CLASSES = load_class_mapping(
    MODEL_DIR / "tamil_classes.json"
)

DEVANAGARI_CLASSES = load_class_mapping(
    MODEL_DIR / "devanagari_classes.json"
)

print(f"Brahmi Classes      : {len(BRAHMI_CLASSES)}")
print(f"Tamil Classes       : {len(TAMIL_CLASSES)}")
print(f"Devanagari Classes  : {len(DEVANAGARI_CLASSES)}")

# ==========================================================
# MODEL SELECTOR
# ==========================================================

def get_model(script):
    """
    Returns the correct OCR model
    and class mapping.
    """

    script = script.lower()

    if script == "brahmi":

        return BRAHMI_MODEL, BRAHMI_CLASSES

    elif script == "tamil":

        return TAMIL_MODEL, TAMIL_CLASSES

    elif script == "devanagari":

        return DEVANAGARI_MODEL, DEVANAGARI_CLASSES

    raise ValueError(
        f"Unsupported Script : {script}"
    )

# ==========================================================
# PAD & RESIZE
# ==========================================================

def pad_and_resize(
    image,
    size=75
):
    """
    Resize character image
    while preserving aspect ratio.
    """

    if len(image.shape) == 3:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    h, w = image.shape

    scale = min(
        (size - 10) / w,
        (size - 10) / h
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    x = (size - new_w) // 2

    y = (size - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas

# ==========================================================
# PREPARE IMAGE
# ==========================================================

def prepare_character(image):
    """
    Prepare image for CNN.
    """

    image = pad_and_resize(image)

    image = image.astype(np.float32)

    image /= 255.0

    image = image.reshape(
        1,
        75,
        75,
        1
    )

    return image

# ==========================================================
# PREDICT SINGLE CHARACTER
# ==========================================================

def recognize_character(image, script):
    """
    Predict a single character.
    """

    model, classes = get_model(script)

    sample = prepare_character(image)

    prediction = model.predict(
        sample,
        verbose=0
    )[0]

    index = int(np.argmax(prediction))

    confidence = float(np.max(prediction))

    # Support both list and dictionary class files
    label = classes[index]

    return label, confidence


# ==========================================================
# PREDICT MULTIPLE CHARACTERS
# ==========================================================

def recognize_characters(character_images, script):
    """
    Predict multiple characters.
    """

    predictions = []

    confidences = []

    for image in character_images:

        label, conf = recognize_character(
            image,
            script
        )

        predictions.append(label)

        confidences.append(conf * 100)

    return predictions, confidences


# ==========================================================
# CONVERT TO STRING
# ==========================================================

def characters_to_text(predictions):
    """
    Convert character list into text.
    """

    if len(predictions) == 0:
        return ""

    return "".join(predictions)


# ==========================================================
# COMPLETE OCR PIPELINE
# ==========================================================

def run_ocr(character_images, script):
    """
    Complete OCR pipeline.
    """

    predictions, confidences = recognize_characters(
        character_images,
        script
    )

    text = characters_to_text(predictions)

    return (
        predictions,
        confidences,
        text
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    img = cv2.imread(
        "char_0.png",
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        print("Test image not found.")

    else:

        for script in [
            "Brahmi",
            "Tamil",
            "Devanagari"
        ]:

            label, confidence = recognize_character(
                img,
                script
            )

            print("----------------------------------")
            print("Script     :", script)
            print("Prediction :", label)
            print("Confidence :", f"{confidence*100:.2f}%")