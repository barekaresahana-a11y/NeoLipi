
# 🏛️ NeoLipi: Multi-Script Ancient Inscription Recognition and Translation System Using Deep Learning

## 📌 Project Overview

**NeoLipi** is a deep learning-based system designed to recognize, digitize, and translate ancient Indian inscriptions written in multiple historical scripts. The project focuses on three scripts: **Tamil, Brahmi, and Devanagari**.

Ancient inscriptions engraved on stones and other historical materials are important sources of India's cultural and linguistic heritage. However, manually reading and interpreting these inscriptions is challenging due to weathering, damaged characters, irregular writing patterns, and variations in ancient scripts.

NeoLipi aims to address these challenges by combining image processing techniques and deep learning models to identify scripts, recognize individual characters, reconstruct digital text, and translate the extracted content into Kannada.

## 🎯 Objectives

- To develop an automated system for recognizing ancient inscriptions.
- To classify inscription images into Tamil, Brahmi, and Devanagari scripts.
- To preprocess inscription images for improved character recognition.
- To recognize characters using deep learning-based recognition models.
- To extract and reconstruct digital text from inscription images.
- To convert recognized inscription text into Kannada.
- To support the digital preservation and accessibility of ancient Indian linguistic heritage.

## ✨ Key Features

- **Multi-Script Classification:** Identifies whether an uploaded inscription belongs to Tamil, Brahmi, or Devanagari.
- **Image Preprocessing:** Applies image enhancement, grayscale conversion, noise removal, thresholding, and binarization techniques.
- **Character Recognition:** Uses deep learning-based models to identify characters from inscription images.
- **Character Segmentation:** Separates inscription images into individual character regions for recognition.
- **Text Extraction:** Converts recognized characters into digital text.
- **Kannada Translation:** Converts the recognized text into Kannada using the Lipi Lekhika library.
- **Interactive User Interface:** Provides an interface for uploading inscription images and viewing processing results.
- **Manual Segmentation Support:** Allows users to adjust character segmentation when necessary.

## 🔄 System Workflow

The NeoLipi system follows a sequence of image processing and deep learning operations.

1. **Image Input:** The user uploads an image of an ancient inscription.
2. **Image Preprocessing:** The input image is enhanced and converted into a suitable format for further processing.
3. **Script Classification:** A CNN-based classification model identifies the script as Tamil, Brahmi, or Devanagari.
4. **Image Binarization:** The inscription is converted into a binary image to distinguish characters from the background.
5. **Character Segmentation:** The inscription is divided into smaller character regions using image segmentation techniques.
6. **Character Recognition:** The corresponding trained recognition model predicts the characters.
7. **Text Reconstruction:** The recognized characters are arranged to generate digital text.
8. **Kannada Translation:** The extracted text is converted into Kannada using Lipi Lekhika.
9. **Result Display:** The interface displays the processing results and translated output.

## 🧠 Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras, CNN |
| Computer Vision | OpenCV |
| Image Processing | NumPy, Pillow |
| Machine Learning | Scikit-learn |
| OCR Techniques | CNN, BiLSTM, CTC-based sequence recognition |
| Translation | Lipi Lekhika |
| User Interface | Streamlit |
| Data Visualization | Matplotlib |
| Development Environment | VS Code, Google Colab |

## 📚 Scripts Supported

### 1. Tamil Script
The system processes Tamil inscription images and applies preprocessing, segmentation, and character recognition techniques to extract digital text.

### 2. Brahmi Script
Brahmi is one of the oldest writing systems used in the Indian subcontinent. NeoLipi uses deep learning-based recognition models and image processing techniques to recognize Brahmi characters from inscription images.

### 3. Devanagari Script
The system processes Devanagari inscription images using trained character recognition models to identify and extract text.

## 🏗️ Model Architecture

NeoLipi uses deep learning and image processing techniques at different stages of the recognition pipeline.

- **CNN-based Script Classifier:** Classifies input images into the supported script categories.
- **Deep Learning Character Recognition:** Trained recognition models identify individual characters.
- **InceptionV3:** Used in the Brahmi character recognition experiments.
- **CNN + BiLSTM + CTC:** Explored for sequence-based OCR and text recognition.
- **Image Segmentation:** Uses image processing techniques to isolate individual characters before recognition.

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.10 or above
- pip package manager
- Git
- Visual Studio Code or another Python IDE

### 1. Clone the Repository

```bash
git clone https://github.com/barekaresahana-a11y/NeoLipi.git
```

### 2. Navigate to the Project Directory

```bash
cd NeoLipi
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / Ubuntu:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 📂 Project Structure

```text
NeoLipi/
│
├── app.py
│
├── models/
│   ├── script_classifier/
│   ├── tamil_model/
│   ├── brahmi_model/
│   └── devanagari_model/
│
├── preprocessing/
│
├── segmentation/
│
├── recognition/
│
├── translation/
│
├── datasets/
│
├── requirements.txt
│
└── README.md
```

*Note: The directory structure above is a suggested organizational structure. Update it to match the actual files in your repository.*

## 📊 Expected Output

The system is designed to provide the following outputs:

- Uploaded ancient inscription image.
- Predicted script category.
- Preprocessed and binarized inscription image.
- Segmented character regions.
- Recognized digital text.
- Kannada translated output.

## 🌍 Applications

- Digital preservation of ancient Indian inscriptions.
- Historical and archaeological research.
- Linguistic and cultural heritage documentation.
- Automated inscription digitization.
- Educational and academic research.
- Making historical text more accessible through regional-language translation.

## 🚀 Future Enhancements

- Improve recognition accuracy for damaged and weathered inscriptions.
- Expand support to additional ancient Indian scripts.
- Improve word-level and sentence-level text reconstruction.
- Develop a larger and more diverse inscription dataset.
- Integrate advanced OCR and language models.
- Support additional regional Indian languages.
- Develop a web-based platform for researchers and historians.

## 👩‍💻 Project Information

**Project Title:** NeoLipi: Multi-Script Ancient Inscription Recognition and Translation System Using Deep Learning

**Project Domain:** Artificial Intelligence, Deep Learning, Computer Vision and Natural Language Processing

**Developed By:** Final Year BE Computer Science and Engineering Students

**Institution:** Vivekananda College of Engineering & Technology, Puttur, Karnataka, India

## 📜 License

This project is currently shared for academic and educational purposes. Please check the repository's license before reusing or redistributing its source code.

## ⭐ Acknowledgement

We acknowledge the datasets, open-source libraries, research contributions, and resources that supported the development of NeoLipi.

**NeoLipi – Preserving Ancient Scripts Through Artificial Intelligence.**
