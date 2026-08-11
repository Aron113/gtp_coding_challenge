"""UBS GCC — one server for all stages.

Exposes:
  /mcp       MCP streamable-HTTP endpoint (Stage 1 "Nursery" tools)
  /solve     Stage 0 adaptive API gateway
  /event     telemetry sink (logged)
  /callback  evaluation result sink (logged)
  /          health check
"""

import ast
import base64
import json
import logging
import math
import os
import re

import cv2
import numpy as np
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gcc")

CHILD_NAME = "Nova"

mcp = MCPServer(
    name="nursery",
    instructions=(
        "Tools for a young assistant: ask for your name, do arithmetic, "
        "and identify or count shapes in base64 PNG images."
    ),
)

# ---------------------------------------------------------------- arithmetic

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_ALLOWED_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def evaluate_arithmetic(expression: str):
    # Keep only characters that can appear in plain arithmetic, so inputs
    # like "What is 2 + 2?" still evaluate.
    cleaned = re.sub(r"[^0-9+\-*/(). ]", " ", expression)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError(f"No arithmetic expression found in: {expression!r}")
    result = _eval_node(ast.parse(cleaned, mode="eval"))
    if isinstance(result, float) and math.isfinite(result) and abs(result - round(result)) < 1e-9:
        return int(round(result))
    return result


# ------------------------------------------------------------------- shapes


def _decode_image(image_base64: str) -> np.ndarray:
    """Base64 PNG -> grayscale image, alpha composited onto white."""
    data = re.sub(r"^data:image/[a-zA-Z+]+;base64,", "", image_base64.strip())
    data = re.sub(r"\s+", "", data)
    raw = base64.b64decode(data + "=" * (-len(data) % 4))
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image data")
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _shape_contours(gray: np.ndarray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(bw)) > 127.0:  # shapes should be the (white) minority
        bw = 255 - bw
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    min_area = max(40.0, 0.0002 * h * w)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def _classify_contour(contour) -> str:
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if peri <= 0 or area <= 0:
        return "circle"
    circularity = 4.0 * math.pi * area / (peri * peri)
    vertices = len(cv2.approxPolyDP(contour, 0.03 * peri, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    extent = area / (rw * rh) if rw > 0 and rh > 0 else 0.0

    # Reference extents (shape area / min-area bounding rect):
    # triangle ~0.5, circle ~pi/4 ~0.785, rectangle ~1.0
    if vertices == 3 or extent < 0.62:
        return "triangle"
    if circularity >= 0.85:
        return "circle"
    if vertices == 4 and extent >= 0.85:
        return "rectangle"
    # Ambiguous: pick the closest reference extent.
    refs = {"triangle": 0.5, "circle": math.pi / 4.0, "rectangle": 1.0}
    return min(refs, key=lambda k: abs(extent - refs[k]))


def analyze_image(image_base64: str):
    contours = _shape_contours(_decode_image(image_base64))
    contours.sort(key=cv2.contourArea, reverse=True)
    labels = [_classify_contour(c) for c in contours]
    return {
        "total": len(labels),
        "rectangle": labels.count("rectangle"),
        "triangle": labels.count("triangle"),
        "circle": labels.count("circle"),
        "shapes": labels,
    }


# -------------------------------------------------------------------- tools


@mcp.tool(description="Returns your own name. Call this when you are asked what your name is.")
def get_name() -> str:
    return CHILD_NAME


@mcp.tool(
    description=(
        "Evaluates an arithmetic expression using +, -, *, / and parentheses, with "
        "standard operator precedence. Examples: '2 + 2', '7 * 8', '10 / 4', "
        "'3 + 4 * 2 - 1'. Returns the numeric result."
    )
)
def calculate(expression: str) -> float:
    return evaluate_arithmetic(expression)


@mcp.tool(
    description=(
        "Identifies the shape drawn in a base64-encoded PNG image. Returns exactly one "
        "lowercase word: 'rectangle', 'triangle', or 'circle'. Pass the raw base64 string."
    )
)
def identify_shape(image_base64: str) -> str:
    result = analyze_image(image_base64)
    if result["total"] == 0:
        raise ValueError("No shape found in the image")
    # Single-shape question: report the largest/first detected shape.
    return result["shapes"][0]


@mcp.tool(
    description=(
        "Counts the shapes in a base64-encoded PNG image. Returns the total number of "
        "shapes plus a per-type breakdown (rectangle / triangle / circle). Pass the raw "
        "base64 string."
    )
)
def count_shapes(image_base64: str) -> dict:
    result = analyze_image(image_base64)
    result.pop("shapes")
    return result


# ------------------------------------------------------- stage 0: /solve

PRIORITY_MAP = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _adapt(body):
    if isinstance(body, dict) and isinstance(body.get("payload"), str):
        try:
            body = json.loads(base64.b64decode(body["payload"]))
        except (ValueError, json.JSONDecodeError):
            body = {}
    if not isinstance(body, dict):
        body = {}
    adapt_input = body.get("adaptInput", body)
    if not isinstance(adapt_input, dict):
        adapt_input = {}
    user = adapt_input.get("user") if isinstance(adapt_input.get("user"), dict) else {}
    meta = adapt_input.get("metadata") if isinstance(adapt_input.get("metadata"), dict) else {}
    action = adapt_input.get("action")
    priority = meta.get("priority")
    return {
        "id": user.get("id"),
        "name": user.get("fullName"),
        "action": action.lower() if isinstance(action, str) else action,
        "priority": PRIORITY_MAP.get(
            priority.upper() if isinstance(priority, str) else priority, 2
        ),
    }


@mcp.custom_route("/solve", methods=["POST"])
async def solve(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse({"adaptOutput": _adapt(body)})


# ------------------------------------------------- telemetry + callbacks


@mcp.custom_route("/event", methods=["POST"])
async def event(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = (await request.body()).decode(errors="replace")
    logger.info("EVENT %s", json.dumps(body) if not isinstance(body, str) else body)
    return JSONResponse({"ok": True})


@mcp.custom_route("/callback", methods=["POST"])
async def callback(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = (await request.body()).decode(errors="replace")
    logger.info("CALLBACK %s", json.dumps(body) if not isinstance(body, str) else body)
    return JSONResponse({"ok": True})


@mcp.custom_route("/", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "name": CHILD_NAME})


# --------------------------------------------------------------------- app

app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
