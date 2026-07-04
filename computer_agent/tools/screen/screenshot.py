"""
Screen tools: take_screenshot, get_screen_size, find_text_on_screen.
These are the observation primitives used by the Vision Agent.
"""

from __future__ import annotations

import base64
import io

from computer_agent.tools.base import RiskLevel, ToolResult, tool


@tool(name="take_screenshot", risk_level=RiskLevel.LOW, category="screen",
      description="Capture a screenshot of the entire screen and return as base64 JPEG.")
def take_screenshot() -> ToolResult:
    """Capture a screenshot of the entire primary screen."""
    try:
        from computer_agent.config import settings
        from PIL import Image
        import pyautogui
        screenshot = pyautogui.screenshot()
        ow, oh = screenshot.width, screenshot.height
        # Point dimensions = the coordinate space pyautogui mouse uses
        point_w, point_h = pyautogui.size()

        max_dim = settings.screenshot_max_dimension
        # Resize target must not exceed point dimensions — avoids >1x scale on Retina
        target_max = min(max_dim, point_w)
        if max(screenshot.width, screenshot.height) > target_max:
            scale = target_max / max(screenshot.width, screenshot.height)
            screenshot = screenshot.resize(
                (int(screenshot.width * scale), int(screenshot.height * scale)),
                resample=Image.Resampling.LANCZOS,
            )

        screenshot = screenshot.convert("RGB")
        buf = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=settings.screenshot_jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return ToolResult.ok(
            output=b64,
            format="base64_jpeg",
            width=screenshot.width,
            height=screenshot.height,
            original_width=ow,
            original_height=oh,
            point_width=point_w,
            point_height=point_h,
        )
    except Exception as e:
        return ToolResult.fail(error=f"Screenshot failed: {e}")


@tool(name="take_region_screenshot", risk_level=RiskLevel.LOW, category="screen",
      description="Capture a screenshot of a specific screen region.")
def take_region_screenshot(x: int, y: int, width: int, height: int) -> ToolResult:
    """
    Capture a region screenshot.

    x: Left coordinate of the region
    y: Top coordinate of the region
    width: Width of the region in pixels
    height: Height of the region in pixels
    """
    try:
        from computer_agent.config import settings
        from PIL import Image
        import pyautogui
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        ow, oh = screenshot.width, screenshot.height

        max_dim = settings.screenshot_max_dimension
        if max(screenshot.width, screenshot.height) > max_dim:
            scale = max_dim / max(screenshot.width, screenshot.height)
            screenshot = screenshot.resize(
                (int(screenshot.width * scale), int(screenshot.height * scale)),
                resample=Image.Resampling.LANCZOS,
            )

        screenshot = screenshot.convert("RGB")
        buf = io.BytesIO()
        screenshot.save(buf, format="JPEG", quality=settings.screenshot_jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return ToolResult.ok(
            output=b64,
            format="base64_jpeg",
            width=screenshot.width,
            height=screenshot.height,
            original_width=ow,
            original_height=oh,
            x=x,
            y=y,
        )
    except Exception as e:
        return ToolResult.fail(error=f"Region screenshot failed: {e}")


@tool(name="get_screen_size", risk_level=RiskLevel.LOW, category="screen",
      description="Get the current screen resolution width and height.")
def get_screen_size() -> ToolResult:
    """Get the primary screen dimensions in pixels."""
    try:
        import pyautogui
        w, h = pyautogui.size()
        return ToolResult.ok(output={"width": w, "height": h})
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="ocr_screenshot", risk_level=RiskLevel.LOW, category="screen",
      description="Extract all visible text from the current screen using OCR.")
def ocr_screenshot() -> ToolResult:
    """Take a screenshot and run OCR to extract all visible text."""
    try:
        import pyautogui
        import pytesseract

        screenshot = pyautogui.screenshot()
        text = pytesseract.image_to_string(screenshot)
        return ToolResult.ok(output=text.strip(), source="ocr")
    except ImportError:
        return ToolResult.fail(error="pytesseract not installed. Run: brew install tesseract")
    except Exception as e:
        return ToolResult.fail(error=f"OCR failed: {e}")


@tool(name="find_text_on_screen", risk_level=RiskLevel.LOW, category="screen",
      description="Find the screen coordinates of text visible on screen using OCR.")
def find_text_on_screen(text: str) -> ToolResult:
    """
    Locate text on the screen and return its bounding box coordinates.

    text: The text string to search for on screen
    """
    try:
        import pyautogui
        import pytesseract

        screenshot = pyautogui.screenshot()
        data = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)

        matches = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if text.lower() in data["text"][i].lower() and int(data["conf"][i]) > 50:
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                matches.append({
                    "text": data["text"][i],
                    "x": x + w // 2,
                    "y": y + h // 2,
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "confidence": data["conf"][i],
                })

        if not matches:
            return ToolResult.fail(error=f"Text '{text}' not found on screen")
        return ToolResult.ok(output=matches, count=len(matches))
    except Exception as e:
        return ToolResult.fail(error=str(e))
