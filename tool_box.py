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
NAME = "toolbox"


def enforce_token_limit(response_text: str) -> str:
    """Ensures responses never exceed 1500 tokens under o200k_base."""
    tokens = ENCODING.encode(response_text)
    if len(tokens) > MAX_TOKENS:
        return ENCODING.decode(tokens[:MAX_TOKENS])
    return response_text


def _process_image_contours(image_base64: str):
    """Decodes base64 PNG, properly handles transparency/background, and returns valid contours."""
    if "base64," in image_base64:
        image_base64 = image_base64.split("base64,")[1]

    # Clean whitespace, newlines, and quotes
    image_base64 = re.sub(r"[^A-Za-z0-9+/=]", "", image_base64)
    image_bytes = base64.b64decode(image_base64)
    
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img_np = np.array(pil_image)

    # 1. Handle Alpha Channel (if image is transparent)
    alpha = img_np[:, :, 3]
    if np.any(alpha < 250):
        # Foreground is where alpha > 128
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        # 2. Standard Grayscale Thresholding for solid backgrounds
        gray = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        
        # Check corner pixels to determine background tone
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        bg_is_light = np.median(corners) > 127
        
        if bg_is_light:
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise (contours with area >= 30 px)
    return [c for c in contours if cv2.contourArea(c) >= 30]


def _classify_contour(contour) -> str:
    """Classifies a contour into triangle, rectangle, or circle."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "circle"
    
    # Polygon approximation
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "rectangle"
    else:
        # Circularity metric: 4 * pi * Area / Perimeter^2
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (peri * peri)
        if circularity >= 0.65 or num_vertices >= 5:
            return "circle"
        return "rectangle"


def solve_shape(image_base64: str) -> str:
    """Identifies the prominent shape in a base64 PNG."""
    try:
        contours = _process_image_contours(image_base64)
        if not contours:
            return "circle"
        # Find dominant shape by area
        primary = max(contours, key=cv2.contourArea)
        return _classify_contour(primary)
    except Exception:
        return "circle"


def solve_shape_count(image_base64: str) -> int:
    """Counts shapes in a base64 PNG."""
    try:
        contours = _process_image_contours(image_base64)
        return len(contours)
    except Exception:
        return 0


def solve_arithmetic(text: str) -> int | float:
    """Evaluates arithmetic expressions with mixed operators (+, -, *, /)."""
    # Strip everything except math characters so surrounding words
    # ("What is ... ?") don't break the expression into fragments.
    cleaned = re.sub(r"[^0-9\+\-\*\/\.\(\)]", "", text).strip()
    result = sympy.sympify(cleaned).evalf()
    
    # Return int if it is an exact integer, else float. sympy.Float must be
    # converted to a native Python float first - round() on a sympy.Float
    # returns another sympy.Float, which isn't JSON-serializable and gets
    # silently stringified (e.g. 52.5 -> the quoted text "52.500000").
    result = float(result)
    if result.is_integer():
        return int(result)
    return round(result, 6)


def answer_question(payload: Any) -> Any:
    """
    Main dispatch function for Stage 1 Nursery questions.
    Accepts text strings, dictionaries, or JSON payloads.
    """
    if payload is None:
        return NAME

    # Extract query text if payload is a dict / JSON object
    if isinstance(payload, dict):
        text = str(
            payload.get("question")
            or payload.get("prompt")
            or payload.get("text")
            or payload.get("message")
            or payload.get("query")
            or payload.get("input")
            or ""
        )
        if "image" in payload or "image_base64" in payload:
            b64 = payload.get("image") or payload.get("image_base64")
            if "how many" in text.lower() or "count" in text.lower():
                return solve_shape_count(b64)
            return enforce_token_limit(solve_shape(b64))
    else:
        text = str(payload)

    text_lower = text.lower()

    # 1. Name query
    if "your name" in text_lower or "what is your name" in text_lower or "who are you" in text_lower:
        return enforce_token_limit(NAME)

    # 2. Shape / Shape Count query with base64 embedded in prompt
    b64_match = re.search(r"(?:iVBORw0KGgo[A-Za-z0-9+/=]+|[A-Za-z0-9+/]{60,}={0,2})", text)
    if b64_match or "shape" in text_lower:
        if b64_match:
            b64_data = b64_match.group(0)
            if any(k in text_lower for k in ["count", "how many", "number of", "total"]):
                return solve_shape_count(b64_data)
            return enforce_token_limit(solve_shape(b64_data))

    # 3. Arithmetic / Sums query (+, -, *, /)
    if any(op in text for op in ["+", "-", "*", "/", "×", "÷"]) or re.search(r"\d", text):
        try:
            return solve_arithmetic(text)
        except Exception:
            pass

    # Default fallback
    return enforce_token_limit(NAME)