import base64
import io
import re
from typing import Any
import cv2
import numpy as np
from PIL import Image
import sympy
import tiktoken

# Encoding requirement from spec: o200k_base
ENCODING = tiktoken.get_encoding("o200k_base")
MAX_TOKENS = 1500


def enforce_token_limit(response_text: str) -> str:
    """Ensures responses never exceed 1500 tokens under o200k_base."""
    tokens = ENCODING.encode(response_text)
    if len(tokens) > MAX_TOKENS:
        return ENCODING.decode(tokens[:MAX_TOKENS])
    return response_text


def _process_image_contours(image_base64: str):
    """Decodes base64 PNG and returns valid contours."""
    # Strip any data URI prefix if present
    if "base64," in image_base64:
        image_base64 = image_base64.split("base64,")[1]

    # Clean whitespace/newlines
    image_base64 = re.sub(r"\s+", "", image_base64)

    image_bytes = base64.b64decode(image_base64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = np.array(pil_image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Threshold image to isolate shapes
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Invert if background was detected as foreground
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise
    return [c for c in contours if cv2.contourArea(c) > 40]


def _classify_contour(contour) -> str:
    """Classifies a contour into triangle, rectangle, or circle."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "rectangle"
    else:
        # Check circularity metric: 4 * pi * Area / Perimeter^2
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
        if circularity > 0.60 or num_vertices > 4:
            return "circle"
        return "rectangle"


def solve_shape(image_base64: str) -> str:
    """Identifies the prominent shape in a base64 PNG."""
    try:
        contours = _process_image_contours(image_base64)
        if not contours:
            return "rectangle"
        primary = max(contours, key=cv2.contourArea)
        return _classify_contour(primary)
    except Exception:
        return "rectangle"


def solve_shape_count(image_base64: str) -> int:
    """Counts shapes in a base64 PNG."""
    try:
        contours = _process_image_contours(image_base64)
        return len(contours)
    except Exception:
        return 0


def solve_arithmetic(text: str) -> int | float:
    """Evaluates arithmetic expressions with mixed operators (+, -, *, /)."""
    # Extract candidate mathematical substrings
    # Matches patterns like "2 + 2", "15 * (3 + 4) / 2", etc.
    math_match = re.search(r"[\d\.\s\+\-\*\/\(\)]+", text)
    if math_match:
        expr = math_match.group(0).strip()
    else:
        expr = text.strip()

    # Clean non-math characters
    cleaned = re.sub(r"[^0-9\+\-\*\/\.\(\) ]", "", expr).strip()
    result = sympy.sympify(cleaned).evalf()
    
    # Return int if it is an exact integer, else float
    if float(result).is_integer():
        return int(result)
    return float(result)


def answer_question(payload: Any) -> Any:
    """
    Main dispatch function for Stage 1 Nursery questions.
    Accepts text strings, dictionaries, or JSON payloads.
    """
    if payload is None:
        return "Pip"

    # Extract query text if payload is a dict / JSON object
    if isinstance(payload, dict):
        text = str(
            payload.get("question")
            or payload.get("prompt")
            or payload.get("text")
            or payload.get("message")
            or payload.get("query")
            or ""
        )
        # Check if direct base64 image or shape payload is provided in dict
        if "image" in payload or "image_base64" in payload:
            b64 = payload.get("image") or payload.get("image_base64")
            if "count" in text.lower():
                return solve_shape_count(b64)
            return solve_shape(b64)
    else:
        text = str(payload)

    text_lower = text.lower()

    # 1. Name query
    if "your name" in text_lower or "what is your name" in text_lower or "who are you" in text_lower:
        return enforce_token_limit("Pip")

    # 2. Shape / Shape Count query with base64 embedded in prompt
    # Long base64 PNG strings typically start with iVBORw0KGgo
    b64_match = re.search(r"(?:iVBORw0KGgo[A-Za-z0-9+/=]+|[A-Za-z0-9+/]{80,}={0,2})", text)
    if b64_match or "shape" in text_lower:
        if b64_match:
            b64_data = b64_match.group(0)
            if "how many" in text_lower or "count" in text_lower:
                return solve_shape_count(b64_data)
            return enforce_token_limit(solve_shape(b64_data))

    # 3. Arithmetic / Sums query (+, -, *, /)
    if any(op in text for op in ["+", "-", "*", "/"]) or re.search(r"\d", text):
        try:
            return solve_arithmetic(text)
        except Exception:
            pass

    # Default fallback
    return enforce_token_limit("Pip")