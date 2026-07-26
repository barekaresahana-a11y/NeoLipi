import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from segmentation.character_segmentation import segment_characters
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

IMG_SIZE = 224

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

@st.cache_resource
def load_model_script():
    return load_model("script_classifier.keras")

script_model = load_model_script()
# -------------------------------------------------
# LOAD OCR MODELS
# -------------------------------------------------

@st.cache_resource
def load_ocr_models():

    devanagari_model = load_model("devanagari_ocr.keras")

    brahmi_model = load_model("brahmi_ocr.keras")

    tamil_model = load_model("tamil_ocr.keras")

    return devanagari_model, brahmi_model, tamil_model


devanagari_model, brahmi_model, tamil_model = load_ocr_models()

CLASS_NAMES = [
    "Brahmi",
    "Devanagari",
    "Tamil"
]
def load_labels(file):

    with open(file, "r") as f:
        return [line.strip() for line in f]


devanagari_labels = load_labels("devanagari_labels.txt")

brahmi_labels = load_labels("brahmi_labels.txt")

tamil_labels = load_labels("tamil_labels.txt")
st.write("Devanagari Classes :", len(devanagari_labels))
st.write("Brahmi Classes :", len(brahmi_labels))
st.write("Tamil Classes :", len(tamil_labels))

st.write("Devanagari Model:", devanagari_model)
st.write("Tamil Model:", tamil_model)
st.write("Brahmi Model:", brahmi_model)

# -------------------------------------------------
# PREPROCESS IMAGE
# -------------------------------------------------

def preprocess_stone(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    gray = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21, # Reduced block size prevents thick merged letters
    8   # Fine-tuned constant C
)

    kernel = np.ones((2,2), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return gray, binary


# -------------------------------------------------
# CLEAN CHARACTER
# -------------------------------------------------

def clean_character(roi):

    roi = cv2.equalizeHist(roi)

    roi = cv2.GaussianBlur(
        roi,
        (3,3),
        0
    )

    _, roi = cv2.threshold(
        roi,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    roi = cv2.resize(
        roi,
        (IMG_SIZE, IMG_SIZE)
    )

    return roi


# -------------------------------------------------
# SEGMENT IMAGE
# -------------------------------------------------

def segment(image):

    gray, binary = preprocess_stone(image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    parts = []
    centers_y = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 50:
            continue

        if area > 5000:
            continue

        if w < 5 or h < 10:
            continue

        roi = gray[y:y+h, x:x+w]

        roi = clean_character(roi)

        parts.append(roi)

        centers_y.append(y + h//2)

    return parts, centers_y


# -------------------------------------------------
# TEXT STRUCTURE CHECK
# -------------------------------------------------

def has_text_structure(centers_y):

    if len(centers_y) < 5:
        return False

    hist, _ = np.histogram(
        centers_y,
        bins=10
    )

    return np.sum(hist > 2) >= 2


# -------------------------------------------------
# TEXT LIKE CHECK
# -------------------------------------------------

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

    valid = np.sum(
        (ratios > 0.2) &
        (ratios < 5)
    )

    return valid > len(parts) * 0.5
    # -------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------

def classify(image):

    gray, binary = preprocess_stone(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rgb = cv2.resize(rgb, (224, 224))

    rgb = rgb.astype(np.float32)

    rgb = preprocess_input(rgb)

    rgb = np.expand_dims(rgb, axis=0)

    prediction = script_model.predict(rgb, verbose=0)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    script = CLASS_NAMES[index]

    return script, confidence
# -------------------------------------------------
# OCR RECOGNITION
# -------------------------------------------------

def recognize_character(character_img, detected_script):

    char = cv2.resize(character_img, (96, 96))

    if len(char.shape) == 2:
        char = cv2.cvtColor(char, cv2.COLOR_GRAY2RGB)

    char = mobilenet_preprocess(char.astype(np.float32))
    char = np.expand_dims(char, axis=0)

    # Select model
    if detected_script == "Tamil":
        model = tamil_model
        labels = tamil_labels

    elif detected_script == "Brahmi":
        model = brahmi_model
        labels = brahmi_labels

    else:
        model = devanagari_model
        labels = devanagari_labels

    # ---------- DEBUG ----------
    st.write("Model:", model)
    st.write("Shape:", char.shape)
    st.write("Type:", char.dtype)
    st.write("Min:", float(np.min(char)))
    st.write("Max:", float(np.max(char)))
    st.write("NaN:", np.isnan(char).any())
    # ---------------------------

    prediction = model.predict(char, verbose=0)

    index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return labels[index], confidence
    # -------------------------------------------------
    # EXTRACT CHARACTERS
    # -------------------------------------------------

# ✅ CORRECTED:
def extract_characters(image, script):
    gray, binary = preprocess_stone(image)
    characters, boxes = segment_characters(binary, script)
    return gray, binary, characters, boxes
# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------

st.title("🪨 NeoLipi - Ancient Script Classifier")

uploaded_file = st.file_uploader(
    "Upload Stone Inscription Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if st.button("Analyze"):

        with st.spinner("Analyzing..."):

            result, confidence = classify(img)

            gray, binary, characters, boxes = extract_characters(img, result)

        st.success(f"Detected Script : {result}")

        st.write(f"Confidence : {confidence*100:.2f}%")

        # Draw bounding boxes
        boxed = img.copy()

        for (x, y, w, h) in boxes:

            cv2.rectangle(
                boxed,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )

        boxed = cv2.cvtColor(
            boxed,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Detected Characters")

        st.image(
            boxed,
            use_container_width=True
        )

        st.subheader("Binary Image")

        st.image(
            binary,
            use_container_width=True
        )

        st.subheader("Extracted Characters")

        if len(characters) == 0:

            st.warning("No characters detected.")

        else:

            cols = st.columns(6)

            recognized_text = []

            for i, ch in enumerate(characters):

                 label, conf = recognize_character(ch, result)

                 recognized_text.append(label)

                 with cols[i % 6]:

                     st.image(
                         ch,
                         caption=f"Char {i+1}",
                         use_container_width=True
                    )

                     st.caption(label)

            st.info(
              f"Total Characters Detected : {len(characters)}"
           )
            st.subheader("Recognized Text")

            st.write(" ".join(recognized_text))