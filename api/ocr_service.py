import io
import os
import sys
import numpy as np
from PIL import Image
import easyocr

class OCRService:
    """
    Singleton Optical Character Recognition (OCR) Engine for MemeGuard AI.
    Extracts embedded textual memes/captions using EasyOCR (CRAFT + CRNN).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OCRService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            print("[OCRService] Initializing EasyOCR Reader (English, CPU mode)...")
            self.reader = easyocr.Reader(['en'], gpu=False)
            self._initialized = True
            self.is_available = True
            self.error_message = None
            print("[OCRService] EasyOCR Reader initialized successfully!")
        except Exception as e:
            self.reader = None
            self.is_available = False
            self.error_message = str(e)
            print(f"[OCRService ERROR] Failed to initialize EasyOCR: {e}")

    def extract_text(self, image_bytes: bytes) -> dict:
        """
        Extracts embedded text from raw image bytes.
        Returns a dictionary with success status, clean extracted text, and detection flags.
        """
        if not self.is_available or self.reader is None:
            raise RuntimeError(f"OCR service is unavailable: {self.error_message}")

        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Uploaded image file is empty.")

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format or corrupted file: {e}")

        # Optimize resolution for fast OCR inference if image is excessively large
        max_dim = 1600
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        img_np = np.array(pil_img)

        try:
            # detail=0 returns a list of text strings in top-to-bottom reading order
            raw_lines = self.reader.readtext(img_np, detail=0)
            cleaned_lines = [line.strip() for line in raw_lines if line and line.strip()]
            full_text = " ".join(cleaned_lines)
            has_text = len(full_text) > 0

            return {
                "success": True,
                "extracted_text": full_text,
                "text_detected": has_text,
                "raw_lines": cleaned_lines,
                "message": "Text extraction completed successfully." if has_text else "No readable text detected in image."
            }
        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {e}")


def get_ocr_service() -> OCRService:
    return OCRService()
