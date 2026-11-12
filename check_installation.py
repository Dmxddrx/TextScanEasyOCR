
import torch
import torchvision
import easyocr
import cv2
import pymupdf
import numpy as np

print("=" * 45)
print("       TEXT SCAN OCR - SYSTEM CHECK")
print("=" * 45)

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("EasyOCR:", easyocr.__version__)
print("OpenCV:", cv2.__version__)
print("PyMuPDF:", pymupdf.__version__)
print("NumPy:", np.__version__)

print("CUDA available:", torch.cuda.is_available())

print("\nAll basic imports successful.")
