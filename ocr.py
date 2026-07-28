from paddleocr import PaddleOCR
import os

# Load PaddleOCR model only once
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)


def recognize_text(image_path):
    """
    Recognizes text from a single image.

    Parameters:
        image_path (str)

    Returns:
        str
    """

    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        return ""

    result = ocr.ocr(image_path, cls=True)
    print("OCR Result:", result)

    extracted_text = []

    if result is None:
        return ""

    for line in result[0]:
        text = line[1][0]
        confidence = line[1][1]

        print(f"{text}   ({confidence:.2f})")

        extracted_text.append(text)

    return " ".join(extracted_text)