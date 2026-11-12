
import cv2
import numpy as np


def order_points(points):

    rect = np.zeros((4, 2), dtype="float32")

    total = points.sum(axis=1)

    rect[0] = points[np.argmin(total)]
    rect[2] = points[np.argmax(total)]

    difference = np.diff(points, axis=1)

    rect[1] = points[np.argmin(difference)]
    rect[3] = points[np.argmax(difference)]

    return rect


def four_point_transform(image, points):

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    width = int(max(width_a, width_b))
    height = int(max(height_a, height_b))

    if width < 100 or height < 100:
        return image

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height)
    )

    return warped


def detect_document(image):

    original = image.copy()

    ratio = image.shape[0] / 900.0

    resized = cv2.resize(
        image,
        (int(image.shape[1] / ratio), 900)
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours[:10]:

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) == 4:

            area = cv2.contourArea(approx)

            if area < resized.shape[0] * resized.shape[1] * 0.15:
                continue

            points = approx.reshape(4, 2).astype(
                "float32"
            )

            points *= ratio

            return four_point_transform(
                original,
                points
            )

    print("Document boundary not detected. Using original frame.")

    return original


def enhance_document(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        5,
        7,
        21
    )

    return enhanced


def process_document(image):

    corrected = detect_document(image)

    enhanced = enhance_document(corrected)

    return corrected, enhanced
