"""
===========================================================
NeoLipi Segmentation V2
Line Segmentation
===========================================================
"""

import cv2
import numpy as np


# ==========================================================
# LINE SEGMENTATION
# ==========================================================

def segment_lines(binary,
                  min_line_height=8,
                  gap_threshold=3):
    """
    Segment a binary inscription image into individual text lines.

    Parameters
    ----------
    binary : ndarray
        Binary image (characters should be white on black)

    Returns
    -------
    list
        List of (line_image, y_offset)
    """

    # Convert to grayscale if required
    if len(binary.shape) == 3:
        gray = cv2.cvtColor(
            binary,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = binary.copy()

    # Ensure foreground is white
    _, bw = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Horizontal projection
    projection = np.sum(bw > 0, axis=1)

    lines = []

    start = None
    empty_count = 0

    for y in range(len(projection)):

        if projection[y] > 0:

            if start is None:
                start = y

            empty_count = 0

        else:

            if start is not None:

                empty_count += 1

                if empty_count >= gap_threshold:

                    end = y - gap_threshold + 1

                    if end - start >= min_line_height:

                        top = max(0, start - 5)
                        bottom = min(bw.shape[0], end + 5)

                        line = bw[top:bottom, :]

                        lines.append(
                            (
                                line,
                                start
                            )
                        )

                    start = None
                    empty_count = 0

    # Last line
    if start is not None:

        end = bw.shape[0]

        if end - start >= min_line_height:

            lines.append(
                (
                    bw[start:end, :],
                    start
                )
            )

    # If no line detected,
    # use the whole image as one line.
    if len(lines) == 0:

        lines.append(
            (
                bw,
                0
            )
        )

    return lines


# ==========================================================
# DEBUG
# ==========================================================

if __name__ == "__main__":

    image = cv2.imread(
        "binary.png",
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print("binary.png not found")

        exit()

    lines = segment_lines(image)

    print("Detected Lines:", len(lines))

    for i, (line, y) in enumerate(lines):

        cv2.imshow(
            f"Line {i+1}",
            line
        )

    cv2.waitKey(0)

    cv2.destroyAllWindows()