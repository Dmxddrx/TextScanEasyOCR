
import cv2

from pathlib import Path
from config import CAMERA_URL, CAPTURE_DIR


def main():

    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 50)
    print("       NOTE20 ULTRA CAMERA TEST")
    print("=" * 50)

    print("Camera URL:", CAMERA_URL)

    camera = cv2.VideoCapture(CAMERA_URL)

    if not camera.isOpened():

        print("\nERROR: Cannot connect to camera.")
        print("Check the phone IP and Wi-Fi connection.")

        return

    print("\nCamera connected successfully.")

    print("Press C = Capture image")
    print("Press Q = Quit")

    while True:

        success, frame = camera.read()

        if not success:
            print("Frame reception failed.")
            break

        preview = frame.copy()

        cv2.putText(
            preview,
            "C: Capture | Q: Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Note20 Ultra Camera",
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):

            image_path = CAPTURE_DIR / "test_capture.jpg"

            saved = cv2.imwrite(
                str(image_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 98]
            )

            if saved:
                print("Image saved:", image_path)
            else:
                print("Image save failed.")

        elif key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
