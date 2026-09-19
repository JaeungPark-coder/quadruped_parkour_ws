# quadruped_parkour_ws

Cat-Inspired Gait Reward (CIGR) added on top of Isaac Lab's built-in Unitree
Go2 rough-terrain locomotion task. See the paper this implements:
박상백 외, "고양이 유래 보상 기법을 활용한 4족 보행 로봇의 익스트림 파쿠르 방법",
2025 한국정보기술학회 추계 종합학술대회.

Only the reward function is new (`source/quadruped_parkour/quadruped_parkour/tasks/cigr_go2/`) --
everything else (robot, terrain, PPO hyperparameters) reuses Isaac Lab's
existing `Isaac-Velocity-Rough-Unitree-Go2-v0` task as-is (id confirmed
against the installed Isaac Lab's own
`isaaclab_tasks.manager_based.locomotion.velocity.config.go2` registration).

**2026-09-19: the "Setup" section below used to say this repo wasn't
runnable by itself and needed `isaaclab.sh --new` run first.** That's no
longer true -- the generated scaffold (`scripts/rsl_rl/{train,play}.py`,
`source/quadruped_parkour/pyproject.toml`, etc.) is already committed here
(see git history: "Add IsaacLab external-project scaffold", "Merge IsaacLab
scaffold with reward-logging split"). The task registration step this
section used to describe (manually adding `from . import cigr_go2` to
`tasks/__init__.py`) is also unnecessary and does not match the actual
generated file: `tasks/__init__.py` uses Isaac Lab's own
`import_packages(__name__, blacklist_pkgs)`, which recursively imports every
non-blacklisted sub-package (`cigr_go2` included) and so already runs its
`gym.register(...)` calls with no manual wiring. Confirmed by reading both
files directly, not yet by running training (see the smoke test below,
still unexecuted).

## Setup (run this once, on the machine with Isaac Lab installed)

Nothing to generate -- clone this repo where Isaac Lab can find it and
install the task package:

```bash
cd quadruped_parkour_ws/source/quadruped_parkour
python -m pip install -e .
```

If you're instead starting a NEW project from scratch and want to use this
repo's `cigr_go2/` task as the starting point rather than as a full clone,
`<IsaacLab-install-dir>/isaaclab.sh --new` (External project -> RL library:
rsl_rl -> workflow: manager-based) still generates the same scaffold this
repo already has committed -- copy `cigr_go2/` into the generated
`source/<project>/quadruped_parkour/tasks/` and rely on `import_packages`
to pick it up, same as here.

## What was found reading the reward code, before ever training (2026-09-19)

Two bugs, confirmed by reading `mdp/rewards.py`/`mdp/cigr_math.py` and
verified with synthetic-tensor tests (`tests/test_cigr_math.py`) -- neither
has yet been confirmed against a live training run, since none has been
run yet. See `cigr_math.py`'s and `rewards.py`'s own docstrings for the
full reasoning; short version:

1. **Foot height was plain world Z**, meaningless once this task inherits
   rough terrain -- each env's terrain patch sits at a different origin
   height, so standing normally on a tall sub-terrain read as permanently
   elevated feet. This was silently saturating `cigr_flight_clearance` at
   its upper clip (constant 1.0, zero gradient) and firing
   `cigr_takeoff_height`/jump-intent on ordinary standing. Fixed by
   subtracting each env's terrain-patch origin height
   (`env.scene.env_origins`) -- corrects the per-env constant offset, not
   relief within one env's own patch (a step or ramp under one specific
   foot), which needs a per-foot RayCaster, not added yet.
2. **`cigr_landing_symmetry` was comparing front/rear foot centroids' full
   3D distance**, dominated by the fixed ~0.38m front-rear hip spacing on
   Go2 regardless of gait -- `exp(-10 * 0.38) ~= 0.022`, against the
   paper's own 0.05 weight a ~0.001 contribution to the total reward in
   every case, i.e. no usable gradient (what little existed pointed at
   "pull the feet together", unrelated to landing symmetry). Fixed to
   compare front/rear HEIGHT only, which the fixed body length can't
   dominate.

The reward math is now split into two files specifically so this kind of
bug is testable without Isaac Sim running at all:
`mdp/cigr_math.py` holds the actual formulas (pure `torch`, no isaaclab
import), `mdp/rewards.py` holds the isaaclab-dependent glue (body-id
resolution via `ContactSensor.find_bodies`, reading real simulation
tensors). `tests/test_cigr_math.py` exercises the formulas -- including
both bugs above as explicit regression tests -- against synthetic foot
tensors:

```bash
pip install pytest torch  # if not already present
pytest tests/test_cigr_math.py -v
```

This runs in about a second and needs no Isaac Lab installation. It lives
outside `source/` on purpose -- `quadruped_parkour/__init__.py` eagerly
imports `isaaclab_tasks` -> `isaaclab` -> `pxr`, so even a test file that
only wants `cigr_math.py` gets dragged through that chain if it sits
anywhere under the package tree pytest has to import through to find it
(confirmed: `--import-mode=importlib` did not avoid this either).

## Sanity checks, in order

0. Run `pytest tests/test_cigr_math.py -v` first (above) -- it's the
   fastest possible check and already covers the two known-fixed bugs.
1. Before touching CIGR: confirm the *generated* project itself works by
   training the stock task for a few hundred steps:
   ```bash
   python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 64 --headless --max_iterations 5
   ```
2. Smoke-test the CIGR task the same way -- this is where a shape/indexing
   bug in the new reward would first show up. **Still unexecuted** -- this
   is the first real evidence either the scaffold or the reward wiring
   (not just the pure math tested in step 0) actually runs:
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
   from the paper's extracted PDF text). Watch the default rough-terrain
   task's own penalty terms (`dof_torques_l2`, `dof_acc_l2`,
   `undesired_contacts`, `flat_orientation_l2`) against the CIGR curves too
   -- they penalize exactly the explosive, off-level motion a jump takeoff
   needs, and nothing has yet confirmed which side wins.
4. `python scripts/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-CIGR-Play-v0` to watch the trained gait.

## Known gaps

- **No terrain in this task actually requires jumping.** This task
  inherits `Isaac-Velocity-Rough-Unitree-Go2`'s terrain generator config
  as-is (slopes, stairs, random rough) -- nothing in it has a gap, hurdle,
  or box tall enough to force a jump. CIGR rewards jumping ability; a
  terrain that never requires it mostly measures whether a policy can
  satisfy the reward term while walking normally. Adding gap/hurdle/box
  sub-terrains (Isaac Lab's `terrain_gen` config) with an easy-to-hard
  curriculum is unstarted and is the largest remaining piece of work here
  -- without it, a finished training run demonstrates reward-shaping
  mechanics more than it demonstrates parkour.
- **No per-foot terrain-relative height sensor.** The `env_origins`
  subtraction above (fix 1) corrects the per-env constant offset from
  which sub-terrain patch a robot spawned on; it does not read terrain
  height under each individual foot, so relief within one patch (a stair
  edge, a box corner) still reads as if the ground were flat there. Needed
  before the gap/hurdle terrain above would score correctly.
- **No baseline comparison.** A CIGR-trained policy alone can't show
  whether CIGR contributed anything -- the comparison that matters is the
  same terrain/seed/step budget with and without the CIGR reward terms
  enabled. Not run.
- **`takeoff_force_offset`/`takeoff_force_scale`/`takeoff_height_threshold`
  are placeholders**, standing in for the paper's F_c/z_c constants which
  were not recoverable from the extracted PDF text (see `mdp/rewards.py`'s
  ADJUST comment). Untuned.
