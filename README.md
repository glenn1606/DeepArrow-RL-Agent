# ArrowShooting DRL Environment (Gymnasium)

A lightweight environment for training RL agents to shoot arrows at moving targets. Includes wind, gravity, and resource management. 

## Features
- Moving targets with simple bounce dynamics
- Gravity and wind affecting arrows
- Mana and limited arrows
- Clear episode termination/truncation semantics
- Pygame renderer with HUD and keyboard demo

## Requirements
- Python 3.10+ (recommended)
- Dependencies:
  - numpy>=1.24.0
  - gymnasium>=0.29.0
  - pygame>=2.5.0

Install:
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -U pip
pip install -r requirements.txt
```



## Project Structure
- arrow_env.py — Environment implementation and renderer
- demo.py — Manual control demo
- requirements.txt — Core dependencies

## Quick Start (Manual Demo)
```bash
python demo.py
```

Controls:
- Up/Down: angle (0°..90°)
- Left/Right: power (10..50)
- SPACE: shoot (costs 30 mana)
- R: reset episode
- ESC: quit

Tips:
- You need enough mana (≥ 30) to shoot.
- Wind and gravity alter the trajectory.

## Environment API (Gymnasium)
- Action space: Box(low=[0,10,0], high=[90,50,1], dtype=float64)
  - angle_deg ∈ [0, 90]
  - power ∈ [10, 50]
  - shoot ∈ [0, 1] (treat >0 as fire)
- Observation: Python dict
  - player: {x, y}
  - wind: {x, y}
  - resources: {mana, time_left, arrows_left}
  - targets: list of {pos:{x,y}, vel:{x,y}} for active targets
- Step signature (Gymnasium 0.29+):
  - observation, terminated, truncated, info
  - No reward is returned by the env itself.
- Episode end:
  - terminated: all targets destroyed
  - truncated: time runs out or arrows run out

Reproducibility:
```python
obs, info = env.reset(seed=42)
```

## Rewards (Bring Your Own)
This environment does not return reward. Use a wrapper to compute it from info/step_info. Example sketch:

```python
class RewardedEnv(gym.Wrapper):
    def step(self, action):
        obs, terminated, truncated, info = self.env.step(action)
        si = info.get("step_info", {})
        reward = 0.0
        reward += 100.0 * si.get("targets_hit", 0)        # hit
        reward -= 1.0                                      # step cost
        reward -= 5.0 * si.get("arrows_went_out", 0)       # miss
        return obs, reward, terminated, truncated, info
```

Note: For training libraries (e.g., Stable-Baselines3), you will also want to define `observation_space`. Because `targets` is variable-length, you’ll typically encode a fixed-size representation (e.g., pad to `MAX_TARGETS` or flatten to a fixed Box). This repo currently omits `observation_space` on purpose for students to complete.



