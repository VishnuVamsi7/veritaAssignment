from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from ..env.actions import (
    ACTION_ADD_ELEMENT,
    ACTION_CHANGE_COLOR,
    ACTION_MOVE_ELEMENT,
    ELEMENT_IMAGE,
    ELEMENT_SHAPE,
    ELEMENT_TEXT,
)
from ..env.canvas_env import CanvasEnv
from ..env.reward import calculate_reward_with_diagnostics

DEFAULT_TARGET_PROMPT = (
    "Create a Summer Sale email banner with a headline, "
    "a yellow CTA button, and good contrast"
)

mcp = FastMCP("market-canvas-env")
env = CanvasEnv()
current_observation, _ = env.reset(
    seed=0, options={"target_prompt": DEFAULT_TARGET_PROMPT, "initial_elements": 0}
)
last_step_info: dict[str, Any] = {"step": 0}


def _default_action_payload() -> dict[str, Any]:
    return {
        "action_type": ACTION_ADD_ELEMENT,
        "element_type": ELEMENT_TEXT,
        "target_id": 0,
        "dx": [0],
        "dy": [0],
        "new_x": [0],
        "new_y": [0],
        "color_index": 0,
        "content_index": 0,
    }


def _content_index_for_text(content: str) -> int:
    content_lower = content.strip().lower()
    for idx, candidate in enumerate(env.content_library):
        if candidate.strip().lower() == content_lower:
            return idx
    return 0


def _color_index_for_color(color: str) -> int:
    color_lower = color.strip().lower()
    for idx, candidate in enumerate(env.color_palette):
        if candidate.strip().lower() == color_lower:
            return idx
    return 0


def _command_to_action(command: str, params: dict[str, Any]) -> dict[str, Any]:
    action = _default_action_payload()
    cmd = command.strip().lower()

    if cmd in {"add_text", "add_element"}:
        action["action_type"] = ACTION_ADD_ELEMENT
        element_type = str(params.get("type", "text")).lower()
        if element_type == "shape":
            action["element_type"] = ELEMENT_SHAPE
        elif element_type == "image":
            action["element_type"] = ELEMENT_IMAGE
        else:
            action["element_type"] = ELEMENT_TEXT
        action["content_index"] = _content_index_for_text(str(params.get("content", "")))
        if "color" in params:
            action["color_index"] = _color_index_for_color(str(params["color"]))

    elif cmd == "move_element":
        action["action_type"] = ACTION_MOVE_ELEMENT
        action["target_id"] = int(params.get("id", 0))
        if "new_x" in params or "new_y" in params:
            action["new_x"] = [int(params.get("new_x", 0))]
            action["new_y"] = [int(params.get("new_y", 0))]
        else:
            action["dx"] = [int(params.get("dx", 0))]
            action["dy"] = [int(params.get("dy", 0))]

    elif cmd == "change_color":
        action["action_type"] = ACTION_CHANGE_COLOR
        action["target_id"] = int(params.get("id", 0))
        action["color_index"] = _color_index_for_color(str(params.get("color", "#000000")))
    else:
        raise ValueError(f"Unsupported command: {command}")

    return action


@mcp.tool()
def get_canvas_state() -> dict[str, Any]:
    """Return the current semantic canvas observation."""
    return current_observation


@mcp.tool()
def execute_action(command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a high-level command and advance the environment by one step."""
    global current_observation, last_step_info
    payload = _command_to_action(command, params or {})
    observation, reward, terminated, truncated, info = env.step(payload)
    current_observation = observation
    last_step_info = info
    return {
        "command": command,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "observation": observation,
        "info": info,
    }


@mcp.tool()
def get_current_reward() -> dict[str, Any]:
    """Return current scalar reward and reward diagnostics."""
    diagnostics = calculate_reward_with_diagnostics(current_observation, env.target_prompt)
    return {
        "reward": diagnostics["reward"],
        "diagnostics": {
            "components": diagnostics["components"],
            "warnings": diagnostics["warnings"],
            "contrast_ratios": diagnostics["contrast_ratios"],
            "target_prompt": env.target_prompt,
            "step": last_step_info.get("step", 0),
        },
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
