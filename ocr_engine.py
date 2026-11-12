
import easyocr
import cv2
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models" / "easyocr"


class OCREngine:

    def __init__(self):

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Initializing EasyOCR...")

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=str(MODEL_DIR),
            download_enabled=True
        )

        print("EasyOCR initialized successfully.")

    def recognize(self, image):

        print("Running text recognition...")

        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            batch_size=1,
            decoder='greedy',
            workers=0
        )

        extracted = []

        for bbox, text, confidence in results:

            extracted.append({
                "bbox": bbox,
                "text": text,
                "confidence": float(confidence)
            })

        return extracted
