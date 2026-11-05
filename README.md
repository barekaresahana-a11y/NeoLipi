# NeoLipi: Multi-Script Ancient Inscription Recognition and Translation System Using Deep Learning

## 📌 Project Overview

**NeoLipi** is a deep learning-based system developed to recognize ancient Indian inscriptions written in different scripts and convert them into digital text, along with Kannada translation or character mapping.

Ancient inscriptions are valuable sources of historical, cultural, and linguistic information. However, many inscriptions are difficult to understand because they are written in ancient scripts, have faded over time, and are affected by weathering, cracks, uneven surfaces, and other physical damages.

NeoLipi aims to make these inscriptions more accessible by using image processing, deep learning, Optical Character Recognition (OCR), and language conversion techniques.

The system supports three scripts:

- **Brahmi**
- **Tamil**
- **Devanagari**

It first identifies the script in the uploaded inscription image and then processes it using the corresponding script-specific recognition pipeline. The recognized content is converted into digital text and Kannada output is generated through the available translation or character-mapping mechanism.

---

## 🎯 Problem Statement

Ancient Indian inscriptions are important historical records, but manually reading and interpreting them requires specialized linguistic knowledge. Traditional recognition methods are challenging because inscription images often contain noise, damaged characters, irregular spacing, and variations in writing styles.

There is a need for an automated system that can identify multiple ancient scripts, recognize their characters, generate digital text, and provide Kannada output to improve accessibility and understanding.

---

## 🎯 Objectives

The main objectives of NeoLipi are:

1. To develop an automated system for recognizing ancient inscriptions.
2. To identify whether an uploaded inscription belongs to Brahmi, Tamil, or Devanagari script.
3. To preprocess inscription images and improve their quality for recognition.
4. To segment inscription images into individual characters or text regions.
5. To recognize characters using deep learning-based OCR techniques.
6. To convert recognized characters into machine-readable digital text.
7. To provide Kannada translation or character mapping wherever supported by the script-specific pipeline.
8. To develop a user-friendly interface for uploading images and viewing recognition results.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning model development and training |
| EfficientNetB0 | Multi-script classification |
| Custom CNN | Character recognition for Brahmi and Tamil |
| FastOCR | Devanagari text recognition |
| OpenCV | Image preprocessing and segmentation |
| NumPy | Numerical operations and image processing |
| Matplotlib | Visualization of images and results |
| Scikit-learn | Machine learning utilities and evaluation |
| Streamlit | Web application interface |
| Deep Translator | Tamil-to-Kannada translation |
| Lipi Lekhika | Kannada character conversion where integrated |
| Google Gemini API | Devanagari/Sanskrit-to-Kannada translation in the integrated pipeline |

---

## 🏗️ System Architecture

The NeoLipi system follows a sequential image-processing and recognition pipeline.

**Overall workflow:**

1. **Input Image:** The user uploads an image of an ancient inscription.
2. **Preprocessing:** The image is enhanced to improve visibility and reduce noise.
3. **Script Classification:** EfficientNetB0 identifies the script as Brahmi, Tamil, or Devanagari.
4. **Character Segmentation:** The inscription is divided into suitable character or text regions using script-specific segmentation techniques.
5. **Digital Text Generation:** OCR models recognize the characters and generate digital text.
6. **Kannada Translation / Character Mapping:** The recognized content is converted into Kannada using the corresponding script-specific mechanism.
7. **Output:** The system displays the recognized digital text and the available Kannada result.

---

## 🔍 Major Modules

### 1. Image Input Module

- Allows users to upload ancient inscription images.
- Accepts images for processing through the Streamlit interface.
- Passes the uploaded image to the preprocessing pipeline.

### 2. Image Preprocessing Module

Preprocessing improves the quality of inscription images before recognition.

The techniques used across the script-specific pipelines include:

- Grayscale conversion.
- Image enhancement.
- Noise reduction.
- Contrast enhancement.
- Morphological operations.
- Image binarization.

These operations help improve character visibility, especially when dealing with stone inscriptions that have uneven backgrounds, weathering, and damaged strokes.

### 3. Script Classification Module

NeoLipi uses **EfficientNetB0** to classify the uploaded inscription image into one of three script categories:

- Brahmi
- Tamil
- Devanagari

EfficientNetB0 is a convolutional neural network architecture that uses compound scaling to balance network depth, width, and image resolution.

The model uses transfer learning with pretrained ImageNet weights and is fine-tuned for the three script classes.

The classification module identifies the script and directs the image to the appropriate recognition pipeline.

### 4. Brahmi Recognition Module

The Brahmi recognition pipeline processes ancient Brahmi inscription images through multiple stages.

**Processing steps:**

1. Image enhancement and illumination normalization.
2. Contrast enhancement using CLAHE.
3. Morphological operations for improving inscription strokes.
4. Percentile-based thresholding for image binarization.
5. Noise removal and connected component analysis.
6. Character segmentation using projection profiles, connected components, and skeletonization-based processing.
7. Character recognition using a custom CNN model.
8. Mapping recognized class labels to corresponding Brahmi Unicode information.
9. Generating digital text and Kannada character output wherever the mapping is available.

The Brahmi pipeline also uses character-level confidence scores to indicate the model's prediction confidence.

### 5. Tamil Recognition Module

The Tamil pipeline processes inscription images using image processing and deep learning-based recognition techniques.

**Processing steps:**

1. Image enhancement and noise reduction.
2. Morphological operations to improve character visibility.
3. Image binarization using Otsu's thresholding.
4. Line segmentation using horizontal projection profiles.
5. Character segmentation using contour-based techniques.
6. Character recognition using a custom CNN model.
7. Digital text generation.
8. Tamil-to-Kannada translation using the integrated translation utility.

The translation utility uses Deep Translator with GoogleTranslator as the primary translation mechanism and MyMemory as a fallback.

### 6. Devanagari Recognition Module

The Devanagari pipeline uses a script-specific OCR approach to recognize text from the uploaded inscription image.

**Processing steps:**

1. Image preprocessing.
2. Preparation of the inscription image for OCR.
3. Text recognition using FastOCR.
4. Generation of recognized digital text.
5. Kannada translation through the integrated Sanskrit-to-Kannada translation component.

The translation component uses the Google Gemini API in the integrated pipeline.

### 7. Kannada Translation and Character Mapping Module

The system provides Kannada output using script-specific conversion mechanisms.

- **Brahmi:** Recognized character classes are mapped to available Brahmi Unicode information and corresponding Kannada character representations.
- **Tamil:** Recognized Tamil text is passed through the Tamil-to-Kannada translation utility.
- **Devanagari:** Recognized text is passed through the integrated Sanskrit-to-Kannada translation component.

The conversion process depends on the script and the availability of the corresponding mapping or translation support.

### 8. User Interface Module

NeoLipi uses **Streamlit** to provide a simple and interactive web interface.

The interface allows users to:

- Upload ancient inscription images.
- View the uploaded image.
- Identify the detected script.
- View processed images and recognition results.
- Display generated digital text.
- View the available Kannada translation or character mapping.

---

## 🔄 Working Methodology

The complete working methodology of NeoLipi is:

**Input Image → Image Preprocessing → Script Classification → Script-Specific Segmentation → OCR and Character Recognition → Digital Text Generation → Kannada Translation / Character Mapping → Final Output**

Each script follows its respective recognition and conversion pipeline after classification.

---

## 🧠 Deep Learning Models and Techniques

| Model / Technique | Application |
|---|---|
| EfficientNetB0 | Classification of Brahmi, Tamil, and Devanagari scripts |
| Custom CNN | Character recognition for Brahmi and Tamil |
| FastOCR | Devanagari text recognition |
| Transfer Learning | Fine-tuning pretrained EfficientNetB0 for script classification |
| CLAHE | Contrast enhancement |
| Otsu's Thresholding | Tamil image binarization |
| Percentile-Based Thresholding | Brahmi image binarization |
| Connected Component Analysis | Identifying connected character regions |
| Skeletonization | Supporting Brahmi character segmentation |
| Projection Profiles | Text-line segmentation |
| Contour-Based Segmentation | Tamil character segmentation |

---

## ⭐ Key Features

- **Multi-Script Support:** Supports three scripts: Brahmi, Tamil, and Devanagari.
- **Automatic Script Classification:** Identifies the script using EfficientNetB0.
- **Image Preprocessing:** Improves inscription image quality before recognition.
- **Script-Specific OCR:** Uses appropriate recognition pipelines for different scripts.
- **Character Segmentation:** Separates characters or text regions for recognition.
- **Digital Text Generation:** Converts recognized inscription content into machine-readable text.
- **Kannada Output:** Provides Kannada translation or character mapping through the available conversion modules.
- **Interactive Interface:** Provides a Streamlit-based application for easy usage.

---

## 💻 Installation and Execution

### Prerequisites

- Python 3.10 or compatible Python environment.
- pip package manager.
- Git.
- Required deep learning models and supporting files.
- Internet connection for translation services that use external APIs.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### Step 2: Navigate to the Project Directory

```bash
cd YOUR-REPOSITORY
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the required model files, datasets, mapping files, and API configuration are available before running the application.

### Step 5: Run the Application

```bash
streamlit run app_final5.py
```

Open the local URL displayed in the terminal to access the NeoLipi application.

*Note: If your main application file has a different name, replace `app_final5.py` with the correct filename.*

---

## 📊 Expected Output

The system produces the following results:

1. Detected script name: Brahmi, Tamil, or Devanagari.
2. Preprocessed inscription image.
3. Recognized digital text.
4. Kannada translation or character mapping, depending on the selected script.
5. Recognition confidence information where supported by the pipeline.

The final output helps users access ancient inscription content in a digital format and understand the available Kannada representation.

---

## 🚀 Applications

- Digital preservation of ancient Indian inscriptions.
- Historical and archaeological research.
- Digital documentation of cultural heritage.
- Educational applications for learning ancient Indian scripts.
- Support for researchers working with historical manuscripts and inscriptions.
- Development of multilingual heritage documentation systems.

---

## 🔮 Future Enhancements

- Expand the system to support additional ancient Indian scripts.
- Improve recognition accuracy for heavily damaged and weathered inscriptions.
- Develop larger and more diverse datasets of stone inscription characters.
- Improve character segmentation for connected and irregular inscriptions.
- Enhance recognition using advanced deep learning architectures.
- Improve language conversion using more comprehensive linguistic resources.
- Develop a centralized digital repository of recognized inscriptions.
- Extend the system to support more Indian languages.
- Explore automated historical inscription interpretation and contextual analysis.

---

## 👩‍💻 Project Information

**Project Title:** NeoLipi: Multi-Script Ancient Inscription Recognition and Translation System Using Deep Learning

**Domain:** Artificial Intelligence, Deep Learning, Image Processing, and Natural Language Processing

**Supported Scripts:** Brahmi, Tamil, and Devanagari

**Interface:** Streamlit

**Core Language:** Python

**Project Type:** Final Year Engineering Project

---

## 📌 Conclusion

NeoLipi presents a deep learning-based approach to recognizing ancient Indian inscriptions written in Brahmi, Tamil, and Devanagari scripts. By combining image preprocessing, automatic script classification, character segmentation, OCR, and Kannada language conversion, the system aims to transform historical inscription images into accessible digital content.

The project contributes towards the digital preservation of India's linguistic and cultural heritage by making ancient inscriptions easier to process, recognize, and access through a unified application.

---

**NeoLipi – Preserving Ancient Scripts Through Deep Learning.**
