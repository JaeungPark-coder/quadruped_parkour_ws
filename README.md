# quadruped_parkour_ws

Cat-Inspired Gait Reward (CIGR) added on top of Isaac Lab's built-in Unitree
Go2 rough-terrain locomotion task. See the paper this implements:
박상백 외, "고양이 유래 보상 기법을 활용한 4족 보행 로봇의 익스트림 파쿠르 방법",
2025 한국정보기술학회 추계 종합학술대회.

Only the reward function is new (`source/quadruped_parkour/quadruped_parkour/tasks/cigr_go2/`) --
everything else (robot, terrain, PPO hyperparameters) reuses Isaac Lab's
existing `Isaac-Velocity-Rough-Unitree-Go2-v0` task as-is.

## Setup (run this once, on the machine with Isaac Lab installed)

This repo only contains the CIGR task files -- it's not a runnable project by
itself yet. Isaac Lab's own project generator builds the rest
(`scripts/rsl_rl/{train,play}.py`, `setup.py`, etc.), since that's
version-matched boilerplate best generated fresh rather than hand-copied:

```bash
cd quadruped_parkour_ws
<IsaacLab-install-dir>/isaaclab.sh --new
# When prompted: External project -> RL library: rsl_rl -> workflow: manager-based
# -> project name: quadruped_parkour
```

This creates `source/quadruped_parkour/` and `scripts/`. The `cigr_go2/` task
folder already committed here goes under
`source/quadruped_parkour/quadruped_parkour/tasks/` -- if the generator
created a different `tasks/` layout or placeholder task, move `cigr_go2/`
alongside it (or replace the placeholder).

Then register the task package by adding to
`source/quadruped_parkour/quadruped_parkour/tasks/__init__.py`:
```python
from . import cigr_go2  # noqa: F401
```

## Sanity checks, in order

1. Before touching CIGR: confirm the *generated* project itself works by
   training the stock task for a few hundred steps:
   ```bash
   python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 64 --headless --max_iterations 5
   ```
2. Smoke-test the CIGR task the same way -- this is where a shape/indexing
   bug in the new reward would first show up:
   ```bash
   python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go2-CIGR-v0 --num_envs 64 --headless --max_iterations 5
   ```
3. Once that runs clean, launch a real training run (drop `--max_iterations`,
   raise `--num_envs`) and watch the five `cigr_takeoff_push` /
   `cigr_takeoff_height` / `cigr_takeoff_speed` / `cigr_flight_clearance` /
   `cigr_landing_symmetry` curves in tensorboard (`tensorboard --logdir
   logs/rsl_rl`) -- logged as separate RewTerms (see `cigr_rough_env_cfg.py`)
   specifically so you can see whether one component (e.g. `cigr_takeoff_speed`,
   weight 1.68) is dominating the other four instead of only their pre-summed
   total. This is also where `takeoff_force_offset` / `takeoff_height_threshold`
   in `cigr_rough_env_cfg.py` will likely need tuning (see the ADJUST comments
   there and in `mdp/rewards.py` -- these two constants weren't recoverable
   from the paper's extracted PDF text).
4. `python scripts/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-CIGR-Play-v0` to watch the trained gait.
