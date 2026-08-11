import ast
import base64
import math
import re
import struct
import zlib
from typing import Any

NAME = "ghost"


def _looks_like_base64(value: str) -> bool:
    if not value:
        return False
    stripped = value.strip()
    if stripped.startswith("data:image"):
        return True
    if stripped.startswith("iVBORw"):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9+/=]+", stripped)) and len(stripped) % 4 == 0


def extract_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "prompt", "question", "query", "input", "message", "content", "request"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if any(key in payload for key in ("image", "png", "base64", "data")):
            return "What shape is this?"
        if "payload" in payload and isinstance(payload["payload"], dict):
            return extract_text(payload["payload"])
    return ""


def is_shape_request(payload: Any) -> bool:
    if isinstance(payload, dict):
        if any(key in payload for key in ("image", "png", "base64", "data")):
            return True
        text = extract_text(payload)
        return "shape" in text.lower() and "this" in text.lower()
    return False


def answer_question(payload: Any) -> Any:
    if is_shape_request(payload):
        return classify_shape(payload)

    text = extract_text(payload)
    lower = text.lower()

    if "what is your name" in lower:
        return NAME

    if any(op in text for op in "+-*/"):
        try:
            result = evaluate_expression(text)
            if result is not None:
                return result
        except Exception:
            pass

    if "shape" in lower and "this" in lower:
        return classify_shape(payload)

    return "I don't know yet."


def evaluate_expression(text: str) -> float | int | None:
    expression = text
    if "what is" in expression.lower():
        expression = expression.split("what is", 1)[1]
    expression = expression.strip().rstrip("?").strip()
    if not expression:
        return None

    allowed = set("0123456789+-*/. ()")
    if any(char not in allowed for char in expression):
        return None

    tree = ast.parse(expression, mode="eval")

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return eval_node(node.left) + eval_node(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return eval_node(node.left) - eval_node(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return eval_node(node.left) * eval_node(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return eval_node(node.left) / eval_node(node.right)
        raise ValueError("unsupported expression")

    result = eval_node(tree.body)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def classify_shape(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("image", "png", "base64", "data"):
            value = payload.get(key)
            if value:
                return _classify_base64(value)
        if "payload" in payload and isinstance(payload["payload"], dict):
            return classify_shape(payload["payload"])

    return _classify_base64(payload)


def _classify_base64(value: Any) -> str:
    if value is None:
        return "rectangle"
    if isinstance(value, bytes):
        data = value
    else:
        text = str(value).strip()
        if text.startswith("data:image"):
            if ";base64," in text:
                _, encoded = text.split(";base64,", 1)
                data = base64.b64decode(encoded)
            else:
                data = base64.b64decode(text)
        elif _looks_like_base64(text):
            data = base64.b64decode(text)
        else:
            return "rectangle"

    try:
        pixels, width, height = _decode_png_pixels(data)
    except Exception:
        return "rectangle"

    if width <= 0 or height <= 0:
        return "rectangle"

    foreground: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            pixel = pixels[row][col]
            if _is_foreground(pixel):
                foreground.append((col, row))

    if not foreground:
        return "rectangle"

    xs = [x for x, _ in foreground]
    ys = [y for _, y in foreground]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max_x - min_x + 1
    bbox_h = max_y - min_y + 1
    bbox_area = bbox_w * bbox_h
    fill_ratio = len(foreground) / bbox_area

    perimeter = 0
    for x, y in foreground:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                perimeter += 1
                continue
            if not _is_foreground(pixels[ny][nx]):
                perimeter += 1

    circularity = 4 * math.pi * len(foreground) / (perimeter * perimeter) if perimeter else 0.0

    if circularity > 0.55:
        return "circle"
    if fill_ratio < 0.55:
        return "triangle"
    return "rectangle"


def _is_foreground(pixel: tuple[int, int, int, int]) -> bool:
    r, g, b, a = pixel
    if a < 200:
        return False
    if r > 240 and g > 240 and b > 240:
        return False
    return True


def _decode_png_pixels(data: bytes) -> tuple[list[list[tuple[int, int, int, int]]], int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG")

    width = 0
    height = 0
    compressed = bytearray()
    offset = 8
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_data = data[offset + 8:offset + 8 + length]
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        offset += 12 + length

    if width == 0 or height == 0:
        raise ValueError("Invalid PNG")

    image_data = zlib.decompress(bytes(compressed))
    bytes_per_pixel = 4 if color_type == 6 else 3
    row_bytes = width * bytes_per_pixel
    pixels: list[list[tuple[int, int, int, int]]] = []
    offset = 0
    prev_row = bytearray(row_bytes)
    for _ in range(height):
        filter_type = image_data[offset]
        offset += 1
        raw_row = image_data[offset:offset + row_bytes]
        offset += row_bytes
        row = _reconstruct_scanline(filter_type, raw_row, prev_row, bytes_per_pixel)
        pixels.append(row)
        prev_row = row

    return pixels, width, height


def _reconstruct_scanline(
    filter_type: int,
    raw_row: bytes,
    previous_row: bytearray,
    bytes_per_pixel: int,
) -> list[tuple[int, int, int, int]]:
    row_len = len(raw_row)
    recon = bytearray(row_len)
    if filter_type == 0:
        recon[:] = raw_row
    elif filter_type == 1:
        for i in range(row_len):
            left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            recon[i] = (raw_row[i] + left) & 0xFF
    elif filter_type == 2:
        for i in range(row_len):
            up = previous_row[i] if len(previous_row) else 0
            recon[i] = (raw_row[i] + up) & 0xFF
    elif filter_type == 3:
        for i in range(row_len):
            left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            up = previous_row[i] if len(previous_row) else 0
            recon[i] = (raw_row[i] + ((left + up) // 2)) & 0xFF
    elif filter_type == 4:
        for i in range(row_len):
            left = recon[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
            up = previous_row[i] if len(previous_row) else 0
            up_left = previous_row[i - bytes_per_pixel] if i >= bytes_per_pixel and len(previous_row) else 0
            pa = abs(left - up)
            pb = abs(left - up_left)
            pc = abs(up - up_left)
            if pa <= pb and pa <= pc:
                predictor = left
            elif pb <= pc:
                predictor = up_left
            else:
                predictor = up
            recon[i] = (raw_row[i] + predictor) & 0xFF
    else:
        raise ValueError("Unsupported PNG filter")

    pixels = []
    for i in range(0, len(recon), bytes_per_pixel):
        if bytes_per_pixel == 3:
            pixels.append((recon[i], recon[i + 1], recon[i + 2], 255))
        else:
            pixels.append((recon[i], recon[i + 1], recon[i + 2], recon[i + 3]))
    return pixels
