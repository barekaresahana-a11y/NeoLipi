import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 128

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():
    tamil = load_model("tamil_vs_all.h5")
    dev = load_model("dev_vs_all.h5")
    brahmi = load_model("brahmi_vs_all.h5")
    return tamil, dev, brahmi

tamil_model, dev_model, brahmi_model = load_models()


# -------------------------
# PREPROCESS
# -------------------------
def preprocess_stone(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    return clahe.apply(gray)


# -------------------------
# CLEAN CHARACTER
# -------------------------
def clean_character(roi):
    roi = cv2.equalizeHist(roi)
    roi = cv2.GaussianBlur(roi, (3,3), 0)

    _, roi = cv2.threshold(
        roi, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return cv2.resize(roi, (IMG_SIZE, IMG_SIZE))


# -------------------------
# SEGMENT CHARACTERS
# -------------------------
def segment(image):

    gray = preprocess_stone(image)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    parts = []
    centers_y = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if 100 < area < 50000:
            roi = gray[y:y+h, x:x+w]
            roi = clean_character(roi)

            parts.append(roi)
            centers_y.append(y + h // 2)

    parts = sorted(parts, key=lambda x: x.shape[0]*x.shape[1], reverse=True)[:40]

    return parts, centers_y


# -------------------------
# TEXT STRUCTURE CHECK
# -------------------------
def has_text_structure(centers_y):

    if len(centers_y) < 5:
        return False

    hist, _ = np.histogram(centers_y, bins=10)
    return np.sum(hist > 2) >= 2


# -------------------------
# 🔥 TEXT-LIKE SHAPE CHECK (NEW FIX)
# -------------------------
def is_text_like(parts):

    if len(parts) < 5:
        return False

    ratios = []

    for p in parts:
        h, w = p.shape
        if h == 0:
            continue
        ratios.append(w / h)

    ratios = np.array(ratios)

    valid = np.sum((ratios > 0.2) & (ratios < 5))

    return valid > len(parts) * 0.5


# -------------------------
# CLASSIFICATION
# -------------------------
def classify(image):

    parts, centers_y = segment(image)

    if len(parts) < 2:
        return "Unclear image", 0

    # 🔥 NEW: reject non-text shapes
    if not is_text_like(parts):
        return "Not a script image", 0

    structure_ok = has_text_structure(centers_y)

    tamil_score = 0
    dev_score = 0
    brahmi_score = 0

    valid = 0

    for p in parts:

        p = p / 255.0
        p = p.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        t = tamil_model.predict(p, verbose=0)[0][0]
        d = dev_model.predict(p, verbose=0)[0][0]
        b = brahmi_model.predict(p, verbose=0)[0][0]

        if max(t, d, b) < 0.35:
            continue

        valid += 1

        tamil_score += t
        dev_score += d
        brahmi_score += b

    if valid < 3:
        if not structure_ok:
            return "Not a script image", 0

    total = tamil_score + dev_score + brahmi_score + 1e-8

    tamil_score /= total
    dev_score /= total
    brahmi_score /= total

    max_score = max(tamil_score, dev_score, brahmi_score)

    if max_score < 0.40:
        if not structure_ok:
            return "Not a script image", max_score
        else:
            return "Low confidence script", max_score

    if brahmi_score > max(tamil_score, dev_score):
        return "Brahmi", max_score
    elif dev_score > max(tamil_score, brahmi_score):
        return "Devanagari", max_score
    else:
        return "Tamil", max_score


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("🪨 Stone Inscription Script Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if st.button("Classify Script"):

        result, confidence = classify(img)

        st.success(f"Detected: {result}")
        st.write(f"Confidence: {round(confidence*100,2)}%")

        # optional confidence level
        if confidence > 0.7:
            st.success("High confidence")
        elif confidence > 0.4:
            st.warning("Medium confidence")
        else:
            st.error("Low confidence")