"""
Screen interaction tools: click, double_click, right_click, drag, scroll.
These are the action primitives for GUI automation.
"""

from __future__ import annotations

from computer_agent.tools.base import RiskLevel, ToolResult, tool


@tool(name="click_coordinate", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Click the mouse at the given screen coordinates.")
def click_coordinate(x: int, y: int) -> ToolResult:
    """
    Move mouse to coordinates and left-click.

    x: Horizontal screen coordinate in pixels
    y: Vertical screen coordinate in pixels
    """
    try:
        import pyautogui
        pyautogui.click(x, y)
        return ToolResult.ok(output=f"Clicked at ({x}, {y})")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="double_click_coordinate", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Double-click at the given screen coordinates.")
def double_click_coordinate(x: int, y: int) -> ToolResult:
    """
    Double-click at the specified coordinates.

    x: Horizontal screen coordinate in pixels
    y: Vertical screen coordinate in pixels
    """
    try:
        import pyautogui
        pyautogui.doubleClick(x, y)
        return ToolResult.ok(output=f"Double-clicked at ({x}, {y})")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="right_click_coordinate", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Right-click at the given screen coordinates to open a context menu.")
def right_click_coordinate(x: int, y: int) -> ToolResult:
    """
    Right-click at the specified coordinates.

    x: Horizontal screen coordinate in pixels
    y: Vertical screen coordinate in pixels
    """
    try:
        import pyautogui
        pyautogui.rightClick(x, y)
        return ToolResult.ok(output=f"Right-clicked at ({x}, {y})")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="move_mouse", risk_level=RiskLevel.LOW, category="screen",
      description="Move the mouse cursor to the given coordinates without clicking.")
def move_mouse(x: int, y: int) -> ToolResult:
    """
    Move mouse cursor without clicking.

    x: Horizontal screen coordinate in pixels
    y: Vertical screen coordinate in pixels
    """
    try:
        import pyautogui
        pyautogui.moveTo(x, y, duration=0.3)
        return ToolResult.ok(output=f"Mouse moved to ({x}, {y})")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="drag_mouse", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Click and drag from one coordinate to another.")
def drag_mouse(from_x: int, from_y: int, to_x: int, to_y: int) -> ToolResult:
    """
    Click-drag from source to destination.

    from_x: Starting horizontal coordinate
    from_y: Starting vertical coordinate
    to_x: Destination horizontal coordinate
    to_y: Destination vertical coordinate
    """
    try:
        import pyautogui
        pyautogui.drag(to_x - from_x, to_y - from_y, duration=0.5, button="left")
        return ToolResult.ok(output=f"Dragged from ({from_x},{from_y}) to ({to_x},{to_y})")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="scroll", risk_level=RiskLevel.LOW, category="screen",
      description="Scroll up or down at the current mouse position.")
def scroll(direction: str, amount: int) -> ToolResult:
    """
    Scroll the mouse wheel.

    direction: 'up' or 'down'
    amount: Number of scroll steps (typically 3-10)
    """
    try:
        import pyautogui
        clicks = amount if direction == "up" else -amount
        pyautogui.scroll(clicks)
        return ToolResult.ok(output=f"Scrolled {direction} by {amount}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="get_mouse_position", risk_level=RiskLevel.LOW, category="screen",
      description="Get the current mouse cursor position on screen.")
def get_mouse_position() -> ToolResult:
    """Return current mouse coordinates."""
    try:
        import pyautogui
        pos = pyautogui.position()
        return ToolResult.ok(output={"x": pos.x, "y": pos.y})
    except Exception as e:
        return ToolResult.fail(error=str(e))
