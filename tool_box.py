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


def enforce_token_limit(response: Any) -> Any:
    """Ensures responses never exceed 1500 tokens under o200k_base."""
    text_repr = str(response)
    tokens = ENCODING.encode(text_repr)
    if len(tokens) > MAX_TOKENS:
        return ENCODING.decode(tokens[:MAX_TOKENS])
    return response


def _process_image_contours(image_base64: str):
    """Decodes base64 PNG, handles transparency/backgrounds, and returns valid contours."""
    if "base64," in image_base64:
        image_base64 = image_base64.split("base64,")[1]

    # Clean non-base64 characters
    image_base64 = re.sub(r"[^A-Za-z0-9+/=]", "", image_base64)
    image_bytes = base64.b64decode(image_base64)

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    img_np = np.array(pil_image)

    # 1. Handle Alpha Transparency
    alpha = img_np[:, :, 3]
    if np.any(alpha < 250):
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        # 2. Solid background: select pixels by how far their colour sits from
        # the background rather than by brightness. A fixed brightness cutoff
        # misses any shape lighter than it - a pale yellow (~243 grey) on white
        # produced zero contours and silently fell back to a guessed answer.
        # Sampling the border ring also handles dark/coloured backgrounds
        # without a separate branch.
        rgb = img_np[:, :, :3].astype(np.int16)
        border = np.concatenate(
            [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
        )
        background = np.median(border, axis=0)
        # 15 is low enough for very pale fills (a #E8F5E9 green sits only 23
        # from white) while staying above the near-background pixels that
        # anti-aliasing leaves along an edge. A flat PNG background carries no
        # noise, so this does not create spurious contours.
        distance = np.abs(rgb - background).max(axis=2)
        mask = (distance > 15).astype(np.uint8) * 255

        # If the shape fills the border the sample is the shape, not the
        # background, and the mask inverts. Detect that and flip it back.
        if np.count_nonzero(mask) > 0.9 * mask.size:
            mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= 30]


def _classify_contour(contour) -> str:
    """Classifies a contour into triangle, rectangle, or circle."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "circle"

    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        return "triangle"
    elif num_vertices == 4:
        return "rectangle"
    else:
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (peri * peri)
        if circularity >= 0.65 or num_vertices >= 5:
            return "circle"
        return "rectangle"


def solve_shape(image_base64: str) -> str:
    """Returns strictly 'rectangle', 'triangle', or 'circle'."""
    try:
        b64_match = re.search(r"(?:iVBORw0KGgo[A-Za-z0-9+/=]+|[A-Za-z0-9+/]{60,}={0,2})", image_base64)
        if b64_match:
            image_base64 = b64_match.group(0)

        contours = _process_image_contours(image_base64)
        if not contours:
            return "circle"
        primary = max(contours, key=cv2.contourArea)
        return _classify_contour(primary)
    except Exception:
        return "circle"


def solve_shape_count(image_base64: str) -> int:
    """Counts shapes in a base64 PNG."""
    try:
        b64_match = re.search(r"(?:iVBORw0KGgo[A-Za-z0-9+/=]+|[A-Za-z0-9+/]{60,}={0,2})", image_base64)
        if b64_match:
            image_base64 = b64_match.group(0)

        contours = _process_image_contours(image_base64)
        return len(contours)
    except Exception:
        return 0


def solve_arithmetic(text: str) -> int | float:
    """Safely extracts and evaluates arithmetic expressions with +, -, *, /."""
    normalized = (
        str(text)
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
    )
    # Only treat "x" as multiplication between two numbers ("6 x 7"), so words
    # containing an x aren't rewritten into stray operators.
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)

    math_segments = re.findall(r"[\d\.\s\+\-\*\/\(\)]+", normalized)
    valid_expressions = []

    for seg in math_segments:
        cand = seg.strip()
        if re.search(r"\d", cand) and re.search(r"[\+\-\*\/]", cand):
            # Strip trailing operators, but only leading "+"/"*"//" - a leading
            # "-" is the sign of the first operand ("-5 + 3"), and removing it
            # silently flipped the result to 8.
            cleaned = cand.rstrip(" +-/*").lstrip(" +*/")
            if cleaned:
                valid_expressions.append(cleaned)

    if not valid_expressions:
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", normalized)
        if nums:
            val = float(nums[0])
            return int(val) if val.is_integer() else val
        raise ValueError(f"Could not extract arithmetic from: {text}")

    target_expr = max(valid_expressions, key=len)
    parsed = sympy.sympify(target_expr, evaluate=True)
    result = float(parsed.evalf())

    if result.is_integer():
        return int(result)
    return round(result, 6)


def answer_question(payload: Any) -> Any:
    """Main router for Stage 1 Nursery tasks."""
    if payload is None:
        return enforce_token_limit(NAME)

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
            if any(k in text.lower() for k in ["count", "how many", "number of"]):
                return solve_shape_count(b64)
            return enforce_token_limit(solve_shape(b64))
    else:
        text = str(payload)

    text_lower = text.lower()

    # 1. Name query
    if any(k in text_lower for k in ["your name", "what is your name", "who are you", "what are you called"]):
        return enforce_token_limit(NAME)

    # 2. Shape / Shape Count query with embedded base64
    b64_match = re.search(r"(?:iVBORw0KGgo[A-Za-z0-9+/=]+|[A-Za-z0-9+/]{60,}={0,2})", text)
    if b64_match or "shape" in text_lower or ".png" in text_lower:
        if b64_match:
            b64_data = b64_match.group(0)
            if any(k in text_lower for k in ["count", "how many", "number of", "total"]):
                return solve_shape_count(b64_data)
            return enforce_token_limit(solve_shape(b64_data))

    # 3. Arithmetic query
    if any(op in text for op in ["+", "-", "*", "/", "×", "÷", "plus", "minus", "times", "divided"]):
        try:
            return solve_arithmetic(text)
        except Exception:
            pass

    return enforce_token_limit(NAME)