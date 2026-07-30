"""
===========================================================
NeoLipi Segmentation V2
Preprocessing
===========================================================
"""

import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess inscription image.

    Returns
    -------
    Binary image with characters in white.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image.copy()

    # Denoise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10
    )

    kernel = np.ones((2, 2), np.uint8)

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

    return binary