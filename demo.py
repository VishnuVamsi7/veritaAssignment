from __future__ import annotations

import json
from typing import Any

from src.env.actions import (
    ACTION_ADD_ELEMENT,
    ACTION_CHANGE_COLOR,
    ACTION_MOVE_ELEMENT,
    ELEMENT_SHAPE,
    ELEMENT_TEXT,
)
from src.env.canvas_env import CanvasEnv
from src.core.renderer import CanvasRenderer


def _action(
    action_type: int,
    *,
    element_type: int = ELEMENT_TEXT,
    target_id: int = 0,
    dx: int = 0,
    dy: int = 0,
    color_index: int = 0,
    content_index: int = 0,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "element_type": element_type,
        "target_id": target_id,
        "dx": [dx],
        "dy": [dy],
        "color_index": color_index,
        "content_index": content_index,
    }


def main() -> None:
    env = CanvasEnv()
    observation, info = env.reset(
        seed=7,
        options={
            "initial_elements": 0,
            "target_prompt": (
                "Create a Summer Sale email banner with a headline, "
                "a yellow CTA button, and good contrast"
            ),
        },
    )
    print(f"Initial state (seed={info.get('seed')}):")
    print(json.dumps(observation, indent=2))

    # 7 logical semantic actions tuned for stronger cumulative reward.
    actions: list[tuple[str, dict[str, Any]]] = [
        (
            "Add headline text early",
            _action(
                ACTION_ADD_ELEMENT,
                element_type=ELEMENT_TEXT,
                color_index=0,  # black
                content_index=4,  # Title
            ),
        ),
        (
            "Add CTA text early",
            _action(
                ACTION_ADD_ELEMENT,
                element_type=ELEMENT_TEXT,
                color_index=0,  # black
                content_index=2,  # Button
            ),
        ),
        (
            "Move headline to centered lane (absolute)",
            {
                **_action(ACTION_MOVE_ELEMENT, target_id=0),
                "new_x": [340],
                "new_y": [80],
            },
        ),
        (
            "Move CTA to centered lane (absolute)",
            {
                **_action(ACTION_MOVE_ELEMENT, target_id=1),
                "new_x": [340],
                "new_y": [260],
            },
        ),
        (
            "Toggle CTA color to orange",
            _action(ACTION_CHANGE_COLOR, target_id=1, color_index=4),  # orange
        ),
        (
            "Keep CTA in yellow/orange family (prompt compliance)",
            _action(ACTION_CHANGE_COLOR, target_id=1, color_index=4),  # orange
        ),
        (
            "No-op absolute move to preserve high-reward layout",
            {
                **_action(ACTION_MOVE_ELEMENT, target_id=1),
                "new_x": [340],
                "new_y": [260],
            },
        ),
    ]

    total_reward = 0.0
    for step_num, (label, action) in enumerate(actions, start=1):
        observation, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        print(f"\nStep {step_num}: {label}")
        print(f"Action: {json.dumps(action)}")
        print(f"Incremental Reward: {reward:.4f}")
        print(f"State: {json.dumps(observation, indent=2)}")
        if terminated or truncated:
            print(
                f"Episode ended early (terminated={terminated}, truncated={truncated}) "
                f"at env step {step_info.get('step')}."
            )
            break

    renderer = CanvasRenderer(env.engine)
    renderer.save_to_png("final_canvas.png")
    print("\nSaved render to final_canvas.png")
    print(f"Total Reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()
