# MarketCanvas-Env Writeup

## 1) Reasoning Behind the Chosen Action and State Spaces

### Environment framing
The environment is designed as a deterministic 2D "Mini-Canva" focused on structured layout decisions rather than photorealistic rendering. The goal is to support RL training and LLM tool interaction for marketing-banner construction under explicit prompt constraints.

### State space
The primary observation is a semantic JSON state (`src/env/observation.py`). Each element includes:
- identity and type (`id`, `type`)
- geometry (`bbox`: `x1`, `y1`, `x2`, `y2`)
- style/content (`color`, `text_color`, `content`)
- spatial relations (`is_centered`, `overlaps_with`)

This representation was chosen because it is:
- deterministic and easy to debug
- compact for rollouts
- directly usable for rule-based rewards
- naturally compatible with MCP tool responses

As an optional visual channel, `CanvasRenderer.render_rgb_array()` in `src/core/renderer.py` provides RGB observations for future multimodal/VLM experiments.

### Action space
I chose a high-level semantic action space (`src/env/actions.py`, `src/env/canvas_env.py`) with:
- `add_element(type, content)`
- `move_element(id, dx, dy)` and `move_element(id, new_x, new_y)`
- `change_color(id, color)`

Why semantic actions:
- lower branching factor than mouse-level actions
- faster policy learning in early-stage RL
- better alignment with the objective (layout validity, CTA presence, contrast quality)

Low-level computer-use actions are still valuable, but are better suited for a later stage where the target is robust interaction with real product UIs (Figma/Canva/Workspace).

## 2) How the Reward Function Works and Potential Loopholes

The reward is implemented in `src/env/reward.py` and returns a scalar in `[-1.0, 1.0]`.
The environment supports two modes (`src/env/canvas_env.py`):
- `dense` (default): reward is returned every step (useful for shaping during policy learning).
- `terminal`: reward is returned only at episode end (strict interpretation of end-of-episode scoring).

### Reward components
Weighted composition:
- **Constraint satisfaction (0.40):** checks whether required prompt elements are present (headline and CTA/button semantics), and enforces prompt color constraints such as "yellow CTA button."
- **Aesthetics (0.25):** penalizes overlap clutter and rewards horizontal centering.
- **Accessibility (0.25):** computes WCAG contrast ratio and rewards text whose contrast is `>= 4.5`.
- **Anti-hacking (0.10):** penalizes repetitive duplicate clusters to prevent inflated rewards from repeated identical elements.

### Reward-hacking prevention
The anti-hacking term uses element signatures `(type, content, color)` and combines:
- duplication-ratio penalty
- saturating penalty (`tanh`) as duplicate count grows

This prevents the trivial exploit of adding many identical CTA elements for unbounded reward gain.

### Known loopholes and limitations
Heuristic rewards remain gameable. Current weaknesses include:
- **keyword gaming:** placing tokens like "Title" and "Button" can satisfy constraints without good design
- **alignment inflation:** many low-value centered elements can raise aesthetics score
- **simplified background model:** contrast currently assumes a default white background in semantic state
- **coarse color semantics:** "yellow CTA" is approximated via named colors / hex heuristics
- **semantic quality gap:** headline quality is lexical, not intent- or typography-aware

Planned mitigations:
- richer prompt parsing (role-aware constraints for headline/CTA/background color and style)
- spacing/coverage/readability constraints
- edit-efficiency penalties over trajectories
- optional learned reward model or preference model on top of heuristics

## 3) Scaling Question: PPO with 10,000 Parallel VLM Rollouts

At this scale, the dominant issues are systems throughput and data movement rather than per-step environment logic.

### Expected bottlenecks
1. **CPU simulator throughput**  
   Python object overhead for updates, overlap checks, and JSON construction.
2. **Rendering overhead**  
   PIL rendering plus RGB extraction becomes expensive when done every step.
3. **Serialization/IPC cost**  
   Large semantic payloads and especially pixel tensors are expensive to move between workers.
4. **Model inference latency**  
   VLM forward passes dominate runtime and can stall rollout workers.
5. **Memory pressure**  
   Storing long trajectories with image observations can quickly exhaust RAM/VRAM.
6. **Protocol overhead**  
   MCP is excellent for tool interoperability but not ideal in the hot loop for high-throughput RL.

### Redesign for scalable training
1. **Dual interfaces**  
   Keep MCP for interactive agent tooling; use direct Gym/vectorized stepping for PPO.
2. **Vectorized environment core**  
   Represent element state as batched arrays/tensors and vectorize overlap/alignment/contrast computations.
3. **Observation tiering**  
   Use semantic-only rollouts by default; render frames selectively (curriculum or sparse checkpoints).
4. **Distributed actor-learner setup**  
   Asynchronous actor pools + centralized learner; co-locate inference and actors to reduce network latency.
5. **Data compression strategy**  
   Store deltas for semantic state; compress pixel frames and shorten rollout windows for vision-heavy runs.
6. **Strict reproducibility controls**  
   Deterministic per-worker seeding and replayable reset/action schedules for debugging and ablation studies.

## 4) Closing Summary

This implementation prioritizes:
- deterministic environment transitions
- interpretable semantic observations
- practical high-level design actions
- bounded and explainable reward shaping
- native MCP tool-calling integration

It is a solid baseline for iterative RL environment research and can be extended into a high-throughput, multimodal training system with the scaling changes outlined above.
