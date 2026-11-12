
import easyocr
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models" / "easyocr"
CAPTURE_DIR = ROOT / "captures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("       EASYOCR MODEL INITIALIZATION TEST")
print("=" * 50)

print("Model directory:", MODEL_DIR)
print("Device: CPU")

# Initialize the pretrained English OCR model
reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=str(MODEL_DIR),
    download_enabled=True
)

print("\nEasyOCR model initialized successfully!")

# Find a captured test image
images = list(CAPTURE_DIR.glob("*.jpg"))

if images:

    image_path = images[0]

    print("\nTesting image:", image_path.name)

    results = reader.readtext(
        str(image_path),
        detail=1,
        paragraph=False
    )

    print("\nRecognized text:")

    for bbox, text, confidence in results:

        print(
            f"{text} | Confidence: {confidence:.3f}"
        )

    print("\nOCR test completed.")

else:

    print("\nNo test image found.")
    print("Capture an image and run this script again.")
