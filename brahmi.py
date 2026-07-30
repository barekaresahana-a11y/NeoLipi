"""
===========================================================
NeoLipi Segmentation V2
Brahmi Segmentation Pipeline
===========================================================
"""

import cv2

from .preprocessing import preprocess_image
from .line_segmentation import segment_lines
from .character_segmentation import segment_characters


# ==========================================================
# BRAHMI SEGMENTATION PIPELINE
# ==========================================================

def segment_brahmi(image, debug=False):
    """
    Complete Brahmi segmentation pipeline.

    Parameters
    ----------
    image : ndarray
        Input inscription image (BGR or grayscale)

    debug : bool
        If True, returns an image with bounding boxes.

    Returns
    -------
    characters : list
        List of segmented character images

    boxes : list
        Character bounding boxes in original image coordinates

    binary : ndarray
        Preprocessed binary image

    debug_img : ndarray or None
        Binary image with bounding boxes drawn
    """

    # ------------------------------------------
    # Step 1 : Preprocess
    # ------------------------------------------

    binary = preprocess_image(image)

    # ------------------------------------------
    # Step 2 : Line Segmentation
    # ------------------------------------------

    lines = segment_lines(binary)

    all_characters = []
    all_boxes = []

    # ------------------------------------------
    # Step 3 : Character Segmentation
    # ------------------------------------------

    for line_img, y_offset in lines:

        characters, boxes = segment_characters(line_img)

        all_characters.extend(characters)

        # Convert line coordinates to original image coordinates
        for x, y, w, h in boxes:
            all_boxes.append(
                (
                    x,
                    y + y_offset,
                    w,
                    h
                )
            )

    # ------------------------------------------
    # Step 4 : Debug Image
    # ------------------------------------------

    debug_img = None

    if debug:

        debug_img = cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR
        )

        for x, y, w, h in all_boxes:

            cv2.rectangle(
                debug_img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return (
        all_characters,
        all_boxes,
        binary,
        debug_img
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    img = cv2.imread("test.png")

    if img is None:
        print("test.png not found")
        exit()

    chars, boxes, binary, debug = segment_brahmi(
        img,
        debug=True
    )

    print(f"Characters Detected : {len(chars)}")

    cv2.imshow("Binary", binary)

    if debug is not None:
        cv2.imshow("Debug", debug)

    for i, ch in enumerate(chars):
        cv2.imshow(f"Character {i+1}", ch)

    cv2.waitKey(0)
    cv2.destroyAllWindows()