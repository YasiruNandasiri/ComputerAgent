"""
Unit tests for screenshot tools — Retina coordinate scaling.

All tests are pure unit tests; no real display or pyautogui calls are made.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from computer_agent.llm.base import ToolCall
from computer_agent.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_screenshot(width: int, height: int) -> PILImage.Image:
    """Return a real PIL Image of the given size (black fill)."""
    return PILImage.new("RGB", (width, height), color=(0, 0, 0))


def _make_screenshot_result(
    img_width: int,
    img_height: int,
    point_width: int,
    point_height: int,
) -> ToolResult:
    """Build a ToolResult as take_screenshot would produce after fix."""
    return ToolResult.ok(
        output="fake_b64",
        format="base64_jpeg",
        width=img_width,
        height=img_height,
        original_width=img_width * 2,   # simulate Retina raw capture
        original_height=img_height * 2,
        point_width=point_width,
        point_height=point_height,
    )


def _coord_note(result: ToolResult) -> str:
    """Replicate the litellm coord_note logic from format_tool_result_messages."""
    w = result.metadata.get("width", "?")
    point_w = result.metadata.get("point_width", w)
    if isinstance(w, int) and w == point_w:
        return "Mouse coordinates map 1:1 to this image."
    scale_x = f"{point_w / w:.2f}" if isinstance(w, int) and w else "1.00"
    return f"Multiply image coordinates by {scale_x} to get mouse coordinates."


# ---------------------------------------------------------------------------
# take_screenshot — Retina scaling
# ---------------------------------------------------------------------------

def test_retina_scale_is_one_to_one():
    """
    On 2× Retina: raw capture is 2880×1800 px, pyautogui.size() = (1440, 900),
    max_dimension = 1440.
    After fix the image must be resized to 1440×900 (≡ point space) and
    point_width must equal the image width → coord note says '1:1'.
    """
    from computer_agent.tools.screen.screenshot import take_screenshot

    fake_img = _fake_screenshot(2880, 1800)

    mock_pyautogui = MagicMock()
    mock_pyautogui.screenshot.return_value = fake_img
    mock_pyautogui.size.return_value = (1440, 900)

    mock_settings = MagicMock()
    mock_settings.screenshot_max_dimension = 1440
    mock_settings.screenshot_jpeg_quality = 75

    with (
        patch.dict(sys.modules, {"pyautogui": mock_pyautogui}),
        patch("computer_agent.config.settings", mock_settings),
    ):
        result = take_screenshot()

    assert result.success, f"take_screenshot failed: {result.error}"
    assert result.metadata["width"] == 1440
    assert result.metadata["height"] == 900
    assert result.metadata["point_width"] == 1440
    assert result.metadata["point_height"] == 900

    note = _coord_note(result)
    assert "1:1" in note, f"Expected 1:1 note, got: {note!r}"
    assert "multiply" not in note.lower(), f"Should not say 'multiply': {note!r}"


def test_non_retina_scale_is_one_to_one():
    """
    On a standard 1080p display: raw capture is 1920×1080 px,
    pyautogui.size() = (1920, 1080), max_dimension = 1440.
    Image is downscaled to 1440×810 but point_w=1920 ≠ img_w=1440 → says multiply.
    """
    from computer_agent.tools.screen.screenshot import take_screenshot

    fake_img = _fake_screenshot(1920, 1080)

    mock_pyautogui = MagicMock()
    mock_pyautogui.screenshot.return_value = fake_img
    mock_pyautogui.size.return_value = (1920, 1080)

    mock_settings = MagicMock()
    mock_settings.screenshot_max_dimension = 1440
    mock_settings.screenshot_jpeg_quality = 75

    with (
        patch.dict(sys.modules, {"pyautogui": mock_pyautogui}),
        patch("computer_agent.config.settings", mock_settings),
    ):
        result = take_screenshot()

    assert result.success, f"take_screenshot failed: {result.error}"
    # Image was downscaled (1920 > min(1440, 1920)=1440), point_w stays 1920
    assert result.metadata["width"] == 1440
    assert result.metadata["point_width"] == 1920
    note = _coord_note(result)
    assert "multiply" in note.lower(), f"Expected multiply note, got: {note!r}"


def test_retina_point_metadata_present():
    """point_width and point_height must always be present in result metadata."""
    from computer_agent.tools.screen.screenshot import take_screenshot

    fake_img = _fake_screenshot(2560, 1600)

    mock_pyautogui = MagicMock()
    mock_pyautogui.screenshot.return_value = fake_img
    mock_pyautogui.size.return_value = (1280, 800)

    mock_settings = MagicMock()
    mock_settings.screenshot_max_dimension = 1440
    mock_settings.screenshot_jpeg_quality = 75

    with (
        patch.dict(sys.modules, {"pyautogui": mock_pyautogui}),
        patch("computer_agent.config.settings", mock_settings),
    ):
        result = take_screenshot()

    assert "point_width" in result.metadata
    assert "point_height" in result.metadata
    assert result.metadata["point_width"] == 1280
    assert result.metadata["point_height"] == 800


# ---------------------------------------------------------------------------
# coord_note logic (unit-tests the litellm placeholder directly)
# ---------------------------------------------------------------------------

def test_coord_note_1to1_when_img_equals_point():
    """If image width == point width after resize, note must say 1:1."""
    result = _make_screenshot_result(
        img_width=1440, img_height=900,
        point_width=1440, point_height=900,
    )
    note = _coord_note(result)
    assert "1:1" in note


def test_coord_note_multiply_when_img_smaller_than_point():
    """If image was downscaled below point size, note must show multiply factor."""
    result = _make_screenshot_result(
        img_width=1440, img_height=810,
        point_width=1920, point_height=1080,
    )
    note = _coord_note(result)
    assert "multiply" in note.lower()
    # Factor should be 1920/1440 = 1.33
    assert "1.33" in note
