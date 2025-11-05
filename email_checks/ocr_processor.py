"""
OCR processor for extracting text from email screenshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import pytesseract


def extract_text_from_image(image_path: str | Path, preprocess: bool = True) -> dict[str, Any]:
    """
    Extract text from an image using OCR.
    
    Args:
        image_path: Path to image file (.png, .jpg, etc.)
        preprocess: Whether to preprocess image for better OCR results
        
    Returns:
        Dictionary with extracted text and metadata
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Preprocess image if requested
    if preprocess:
        processed_image = preprocess_image_for_ocr(image)
    else:
        processed_image = image
    
    # Perform OCR
    try:
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(processed_image, lang='ron+eng')
        
        # Also get detailed data
        ocr_data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, lang='ron+eng')
        
        # Calculate confidence
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}")
    
    return {
        'text': text.strip(),
        'confidence': avg_confidence,
        'file_path': str(image_path),
        'file_name': image_path.name
    }


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Apply thresholding to get binary image
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional: Morphological operations to clean up
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary


def is_image_file(file_path: str | Path) -> bool:
    """
    Check if file is an image file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is an image
    """
    file_path = Path(file_path)
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    return file_path.suffix.lower() in image_extensions

