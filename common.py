"""
===========================================================
NeoLipi Segmentation v2
Common Utility Functions

Shared by:
    - Brahmi
    - Tamil
    - Devanagari

Author: NeoLipi Project
===========================================================
"""

import cv2
import numpy as np


# ==========================================================
# BORDER REMOVAL
# ==========================================================

def strip_border(binary_img):
    """
    Remove white borders surrounding the inscription.
    """

    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return binary_img

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    return binary_img[y:y+h, x:x+w]


# ==========================================================
# SMALL NOISE REMOVAL
# ==========================================================

def remove_small_noise(binary_img, min_area=30):
    """
    Remove tiny connected components.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_img,
        connectivity=8
    )

    cleaned = np.zeros_like(binary_img)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:

            cleaned[labels == i] = 255

    return cleaned


# ==========================================================
# CLAHE ENHANCEMENT
# ==========================================================

def enhance_contrast(gray):
    """
    Improve stone inscription visibility.
    """

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


# ==========================================================
# BINARIZATION
# ==========================================================

def adaptive_binarize(gray):
    """
    Adaptive threshold suitable for uneven stone surfaces.
    """

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15
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


# ==========================================================
# MAIN TEXT REGION
# ==========================================================

def get_main_text_region(binary):
    """
    Extract the largest text block.
    """

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return binary

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    return binary[y:y+h, x:x+w]


# ==========================================================
# TEXT LINE EXTRACTION
# ==========================================================

def get_text_lines(binary):
    """
    Detect text lines using horizontal projection.

    Returns
    -------
    [
        (line_image, y_offset),
        ...
    ]
    """

    projection = np.sum(binary > 0, axis=1)

    H, W = binary.shape

    threshold = max(5, int(W * 0.01))

    lines = []

    start = None

    for y in range(H):

        if projection[y] > threshold:

            if start is None:
                start = y

        else:

            if start is not None:

                if y - start > 8:

                    y1 = max(0, start - 4)
                    y2 = min(H, y + 4)

                    roi = binary[y1:y2, :]

                    lines.append((roi, y1))

                start = None

    if start is not None:

        y1 = max(0, start - 4)
        y2 = H

        roi = binary[y1:y2, :]

        lines.append((roi, y1))

    return lines
# ==========================================================
# MERGE NEARBY COMPONENTS
# ==========================================================

def merge_components(boxes,
                     x_gap=8,
                     y_overlap_ratio=0.60,
                     height_ratio=0.50):
    """
    Merge fragmented components belonging to one character.

    Parameters
    ----------
    boxes : list
        [(x, y, w, h), ...]

    Returns
    -------
    merged_boxes : list
    """

    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):

        if used[i]:
            continue

        x, y, w, h = boxes[i]

        left = x
        top = y
        right = x + w
        bottom = y + h

        used[i] = True

        changed = True

        while changed:

            changed = False

            for j in range(len(boxes)):

                if used[j]:
                    continue

                x2, y2, w2, h2 = boxes[j]

                overlap = min(bottom, y2 + h2) - max(top, y2)

                overlap = max(0, overlap)

                overlap_ratio = overlap / max(1, min(h, h2))

                gap = max(
                    0,
                    max(x2 - right, left - (x2 + w2))
                )

                h_ratio = min(h, h2) / max(h, h2)

                if (
                    gap <= x_gap and
                    overlap_ratio >= y_overlap_ratio and
                    h_ratio >= height_ratio
                ):

                    left = min(left, x2)
                    top = min(top, y2)
                    right = max(right, x2 + w2)
                    bottom = max(bottom, y2 + h2)

                    w = right - left
                    h = bottom - top

                    used[j] = True
                    changed = True

        merged.append((left, top, right-left, bottom-top))

    return merged


# ==========================================================
# SPLIT WIDE COMPONENT
# ==========================================================

def split_projection(binary_roi,
                     min_width_ratio=1.6):
    """
    Split joined characters using
    vertical projection valleys.
    """

    h, w = binary_roi.shape

    if w < h * min_width_ratio:
        return [binary_roi]

    projection = np.sum(binary_roi > 0, axis=0)

    smooth = cv2.GaussianBlur(
        projection.astype(np.float32),
        (1, 9),
        0
    ).flatten()

    valley = np.argmin(
        smooth[int(w*0.2):int(w*0.8)]
    )

    valley += int(w*0.2)

    if valley < 5 or valley > w - 5:
        return [binary_roi]

    left = binary_roi[:, :valley]
    right = binary_roi[:, valley:]

    if left.shape[1] < 5 or right.shape[1] < 5:
        return [binary_roi]

    return [left, right]


# ==========================================================
# SORT BOXES (READING ORDER)
# ==========================================================

def sort_boxes(boxes,
               line_threshold=20):
    """
    Sort character boxes
    from top-left to bottom-right.
    """

    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])

    lines = []

    current = []

    current_y = boxes[0][1]

    for b in boxes:

        if abs(b[1] - current_y) <= line_threshold:

            current.append(b)

        else:

            lines.append(sorted(current,
                                key=lambda x: x[0]))

            current = [b]

            current_y = b[1]

    if current:
        lines.append(sorted(current,
                            key=lambda x: x[0]))

    ordered = []

    for line in lines:
        ordered.extend(line)

    return ordered


# ==========================================================
# CHARACTER CROPPING
# ==========================================================

def crop_characters(binary,
                    boxes,
                    padding=2):
    """
    Crop character images.
    """

    characters = []

    for x, y, w, h in boxes:

        x1 = max(0, x-padding)
        y1 = max(0, y-padding)

        x2 = min(binary.shape[1], x+w+padding)
        y2 = min(binary.shape[0], y+h+padding)

        roi = binary[y1:y2, x1:x2]

        characters.append(roi)

    return characters


# ==========================================================
# PAD & RESIZE
# ==========================================================

def pad_and_resize(img,
                   size=75):
    """
    Center character
    on square canvas.
    """

    h, w = img.shape

    scale = min(
        (size-10)/w,
        (size-10)/h
    )

    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((size, size),
                      dtype=np.uint8)

    x = (size-new_w)//2
    y = (size-new_h)//2

    canvas[y:y+new_h,
           x:x+new_w] = resized

    return canvas
