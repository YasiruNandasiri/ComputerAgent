"""
Keyboard tools: type_text, press_key, hotkey.
"""

from __future__ import annotations

from computer_agent.tools.base import RiskLevel, ToolResult, tool

# Common key names for reference in prompts
_VALID_KEYS = {
    "enter", "tab", "escape", "esc", "space", "backspace", "delete",
    "up", "down", "left", "right",
    "home", "end", "pageup", "pagedown",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
    "ctrl", "alt", "shift", "command", "win", "option",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
}


@tool(name="type_text", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Type text as keyboard input at the current cursor position.")
def type_text(text: str) -> ToolResult:
    """
    Type text character by character using keyboard simulation.

    text: The string to type. Supports all printable characters.
    """
    try:
        import pyautogui
        # Use pyautogui.write for ASCII, typewrite interval for human-like timing
        pyautogui.typewrite(text, interval=0.04)
        return ToolResult.ok(output=f"Typed: {text[:50]}{'...' if len(text) > 50 else ''}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="type_text_fast", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Type text instantly using clipboard paste (faster than typewrite).")
def type_text_fast(text: str) -> ToolResult:
    """
    Type text by pasting from clipboard (fast, preserves special characters).

    text: The string to type via clipboard paste
    """
    try:
        import sys

        import pyautogui
        import pyperclip

        pyperclip.copy(text)
        # Cmd+V on macOS, Ctrl+V elsewhere
        if sys.platform == "darwin":
            pyautogui.hotkey("command", "v")
        else:
            pyautogui.hotkey("ctrl", "v")

        return ToolResult.ok(output=f"Pasted text ({len(text)} chars)")
    except ImportError:
        # Fallback: regular typewrite
        return type_text(text)
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="press_key", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Press a single keyboard key (e.g. enter, escape, tab, f5).")
def press_key(key: str) -> ToolResult:
    """
    Press a single key.

    key: Key name such as 'enter', 'tab', 'escape', 'f5', 'delete'
    """
    try:
        import pyautogui
        pyautogui.press(key.lower())
        return ToolResult.ok(output=f"Pressed key: {key}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="hotkey", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Press a keyboard hotkey combination (e.g. 'cmd+c', 'ctrl+shift+t').")
def hotkey(keys: str) -> ToolResult:
    """
    Press a hotkey combination.

    keys: Plus-separated key names, e.g. 'ctrl+c', 'command+shift+3', 'alt+f4'
    """
    try:
        import pyautogui
        key_list = [k.strip().lower() for k in keys.split("+")]
        pyautogui.hotkey(*key_list)
        return ToolResult.ok(output=f"Hotkey: {keys}")
    except Exception as e:
        return ToolResult.fail(error=str(e))


@tool(name="clear_and_type", risk_level=RiskLevel.MEDIUM, category="screen",
      description="Select all text in the current field and replace it with new text.")
def clear_and_type(text: str) -> ToolResult:
    """
    Clear the current input field and type new text.

    text: The replacement text to enter
    """
    try:
        import sys

        import pyautogui

        # Select all
        if sys.platform == "darwin":
            pyautogui.hotkey("command", "a")
        else:
            pyautogui.hotkey("ctrl", "a")

        # Replace with new text
        pyautogui.typewrite(text, interval=0.04)
        return ToolResult.ok(output=f"Cleared and typed: {text[:50]}")
    except Exception as e:
        return ToolResult.fail(error=str(e))
