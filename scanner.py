
import cv2
import easyocr

from pathlib import Path
from datetime import datetime

from config import (
    CAMERA_URL,
    MODEL_DIR,
    OUTPUT_DIR,
    OCR_LANGUAGES,
    OCR_GPU,
    OCR_BATCH_SIZE,
    OCR_WORKERS,
    IMAGE_JPEG_QUALITY
)

from processing.image_processor import (
    process_document
)

from document_generator import (
    generate_searchable_pdf,
    generate_txt
)


# ==================================================
# DIRECTORY INITIALIZATION
# ==================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# INITIALIZE OCR
# ==================================================

def initialize_ocr():

    print("\nInitializing EasyOCR...")

    reader = easyocr.Reader(
        OCR_LANGUAGES,
        gpu=OCR_GPU,
        model_storage_directory=str(MODEL_DIR),
        download_enabled=True
    )

    print("EasyOCR initialized successfully.")

    return reader


# ==================================================
# CAMERA CAPTURE
# ==================================================

def capture_pages(session_dir):

    original_dir = session_dir / "original_pages"

    original_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nConnecting to Note20 Ultra...")

    camera = cv2.VideoCapture(
        CAMERA_URL
    )

    if not camera.isOpened():

        raise RuntimeError(
            "Unable to connect to Note20 Ultra camera."
        )

    print("Camera connected successfully.")

    print("\nCONTROLS")
    print("C = Capture page")
    print("F = Finish scanning")
    print("Q = Cancel session")

    captured_pages = []

    cancelled = False

    while True:

        success, frame = camera.read()

        if not success:

            print("Camera frame reception failed.")
            break

        preview = frame.copy()

        message = (
            f"Pages: {len(captured_pages)} | "
            "C: Capture | F: Finish | Q: Cancel"
        )

        cv2.putText(
            preview,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "TextScanEasyOCR - Live Camera",
            preview
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):

            page_number = len(captured_pages) + 1

            image_path = (
                original_dir /
                f"page_{page_number:03d}.jpg"
            )

            saved = cv2.imwrite(
                str(image_path),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    IMAGE_JPEG_QUALITY
                ]
            )

            if saved:

                captured_pages.append(image_path)

                print(
                    f"Page {page_number} captured: "
                    f"{image_path}"
                )

            else:

                print("Image save failed.")

        elif key == ord('f'):

            break

        elif key == ord('q'):

            captured_pages.clear()

            cancelled = True

            break

    camera.release()

    cv2.destroyAllWindows()

    if cancelled:

        print("Scanning cancelled.")
        return []

    return captured_pages


# ==================================================
# PROCESS DOCUMENT PAGES
# ==================================================

def process_pages(reader, captured_pages, session_dir):

    processed_dir = (
        session_dir / "processed_pages"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pages = []

    for index, image_path in enumerate(
        captured_pages,
        start=1
    ):

        print("\n" + "=" * 45)

        print(
            f"PROCESSING PAGE {index}/{len(captured_pages)}"
        )

        original = cv2.imread(
            str(image_path)
        )

        if original is None:

            print("Unable to load captured image.")
            continue

        # Image preprocessing
        corrected, enhanced = process_document(
            original
        )

        colour_path = (
            processed_dir /
            f"page_{index:03d}.jpg"
        )

        enhanced_path = (
            processed_dir /
            f"page_{index:03d}_ocr.jpg"
        )

        cv2.imwrite(
            str(colour_path),
            corrected,
            [cv2.IMWRITE_JPEG_QUALITY, 98]
        )

        cv2.imwrite(
            str(enhanced_path),
            enhanced,
            [cv2.IMWRITE_JPEG_QUALITY, 98]
        )

        # OCR recognition
        print("Running EasyOCR...")

        results = reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            batch_size=OCR_BATCH_SIZE,
            workers=OCR_WORKERS,
            decoder="greedy"
        )

        formatted_results = []

        for bbox, text, confidence in results:

            formatted_results.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(confidence)
            })

        print(
            "Text regions detected:",
            len(formatted_results)
        )

        pages.append({
            "image": colour_path,
            "results": formatted_results
        })

    return pages


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():

    print("=" * 55)

    print("         TEXTSCAN - EASYOCR DOCUMENT SCANNER")

    print("=" * 55)

    # Generate session directory
    session_name = datetime.now().strftime(
        "session_%Y%m%d_%H%M%S"
    )

    session_dir = OUTPUT_DIR / session_name

    session_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize OCR once
    reader = initialize_ocr()

    # Capture document pages
    captured_pages = capture_pages(
        session_dir
    )

    if not captured_pages:

        print("\nNo pages captured.")
        return

    print(
        f"\nTotal captured pages: {len(captured_pages)}"
    )

    # Process captured pages
    pages = process_pages(
        reader,
        captured_pages,
        session_dir
    )

    if not pages:

        print("\nNo pages processed successfully.")
        return

    # Output paths
    pdf_path = session_dir / "document.pdf"

    txt_path = session_dir / "document.txt"

    # Generate searchable PDF
    generate_searchable_pdf(
        pages,
        pdf_path
    )

    # Generate TXT
    generate_txt(
        pages,
        txt_path
    )

    print("\n" + "=" * 55)

    print("          SCANNING COMPLETED")

    print("=" * 55)

    print("\nPDF:", pdf_path)

    print("TXT:", txt_path)

    print("\nProcessed pages:", len(pages))


if __name__ == "__main__":

    main()
