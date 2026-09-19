"""Regression tests for cigr_math.py's pure-tensor reward formulas.

Deliberately does not import rewards.py or anything from isaaclab -- see
cigr_math.py's own docstring for why that matters (rewards.py cannot be
imported at all without a running Isaac Sim process). Every test here
constructs synthetic per-foot tensors directly, foot order [FL, FR, RL, RR].

Lives outside the quadruped_parkour package tree (not under source/) on
purpose: quadruped_parkour/__init__.py -> tasks/__init__.py eagerly imports
isaaclab_tasks -> isaaclab -> pxr, so even loading cigr_math.py by file path
from a test file THAT ITSELF sits inside that package tree still triggers
pytest to import every parent __init__.py first, just to resolve the test
module's own qualified name -- confirmed: `--import-mode=importlib` did not
avoid this either. Living here (no __init__.py chain above this directory)
sidesteps that entirely; cigr_math.py itself imports nothing but torch.

Run with: pytest tests/test_cigr_math.py
"""
import importlib.util
import pathlib

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CIGR_MATH_PATH = (_REPO_ROOT / "source/quadruped_parkour/quadruped_parkour"
                    / "tasks/cigr_go2/mdp/cigr_math.py")

_spec = importlib.util.spec_from_file_location("cigr_math", _CIGR_MATH_PATH)
_cigr_math = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cigr_math)

compute_foot_state = _cigr_math.compute_foot_state
flight_clearance = _cigr_math.flight_clearance
jump_intent = _cigr_math.jump_intent
landing_symmetry = _cigr_math.landing_symmetry
takeoff_height = _cigr_math.takeoff_height
takeoff_push = _cigr_math.takeoff_push
takeoff_speed = _cigr_math.takeoff_speed


def test_compute_foot_state_subtracts_terrain_origin():
    """CONFIRMED regression for the 2026-09-19 world-Z bug: two envs
    standing identically relative to their own terrain patch, but one
    patch sits 0.9m higher than the other, must read the same foot
    height once terrain origin is subtracted -- not one reading ~0.9m
    higher, which is what plain world Z gave before this fix."""
    # (N=2, 4 feet, xyz). Both envs' feet sit 0.03m above their own patch.
    flat_env_z = 0.03
    tall_env_origin_z = 0.90
    foot_pos_w = torch.zeros(2, 4, 3)
    foot_pos_w[0, :, 2] = flat_env_z
    foot_pos_w[1, :, 2] = tall_env_origin_z + flat_env_z
    foot_lin_vel_w = torch.zeros(2, 4, 3)
    foot_force_history_z = torch.zeros(2, 1, 4)
    env_origin_z = torch.tensor([0.0, tall_env_origin_z])

    foot_height, _, _ = compute_foot_state(foot_pos_w, foot_lin_vel_w, foot_force_history_z, env_origin_z)

    assert torch.allclose(foot_height[0], foot_height[1], atol=1e-6)
    assert torch.allclose(foot_height[0], torch.full((4,), flat_env_z), atol=1e-6)


def test_compute_foot_state_force_uses_max_over_history():
    foot_pos_w = torch.zeros(1, 4, 3)
    foot_lin_vel_w = torch.zeros(1, 4, 3)
    # (N=1, history=2, 4 feet), one substep of which reads a negative bounce
    foot_force_history_z = torch.tensor([[
        [10.0, -5.0, 0.0, 30.0],
        [5.0, 40.0, -1.0, 20.0],
    ]])
    env_origin_z = torch.zeros(1)

    _, _, foot_force_z = compute_foot_state(foot_pos_w, foot_lin_vel_w, foot_force_history_z, env_origin_z)

    assert torch.allclose(foot_force_z, torch.tensor([[10.0, 40.0, 0.0, 30.0]]))


def test_jump_intent_requires_two_feet():
    # only ONE foot (index 0) clears both thresholds -- should not count as jump intent
    foot_height = torch.tensor([[0.2, 0.0, 0.0, 0.0]])
    foot_planar_speed = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    assert jump_intent(foot_height, foot_planar_speed, height_threshold=0.1, speed_threshold=0.2).item() == 0.0

    # two feet clear both thresholds -- should count
    foot_height = torch.tensor([[0.2, 0.2, 0.0, 0.0]])
    foot_planar_speed = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    assert jump_intent(foot_height, foot_planar_speed, height_threshold=0.1, speed_threshold=0.2).item() == 1.0


def test_takeoff_push_gated_on_both_rear_feet_contact():
    # rear feet (idx 2,3) both above threshold -> nonzero
    foot_force_z = torch.tensor([[0.0, 0.0, 25.0, 30.0]])
    reward = takeoff_push(foot_force_z, contact_threshold=20.0, force_offset=5.0, force_scale=5.0)
    assert reward.item() > 0.0

    # one rear foot below threshold -> gated to zero regardless of force magnitude
    foot_force_z = torch.tensor([[0.0, 0.0, 5.0, 100.0]])
    reward = takeoff_push(foot_force_z, contact_threshold=20.0, force_offset=5.0, force_scale=5.0)
    assert reward.item() == 0.0


def test_takeoff_height_scales_with_rear_clearance_when_jumping():
    foot_height = torch.tensor([[0.0, 0.0, 0.20, 0.20]])
    jumping = torch.tensor([1.0])
    not_jumping = torch.tensor([0.0])
    assert takeoff_height(foot_height, height_threshold=0.05, jump_intent_value=jumping).item() > 0.0
    assert takeoff_height(foot_height, height_threshold=0.05, jump_intent_value=not_jumping).item() == 0.0


def test_takeoff_speed_capped_at_one():
    jumping = torch.tensor([1.0])
    fast = takeoff_speed(torch.tensor([10.0]), max_horizontal_speed=0.8, jump_intent_value=jumping)
    assert fast.item() == 1.0
    slow = takeoff_speed(torch.tensor([0.4]), max_horizontal_speed=0.8, jump_intent_value=jumping)
    assert 0.0 < slow.item() < 1.0


def test_flight_clearance_saturates_within_band_not_outside_it():
    height_min, height_max = 0.03, 0.12
    # a foot well within the band scores strictly between 0 and 1
    mid = flight_clearance(torch.tensor([[0.07, 0.07, 0.07, 0.07]]), height_min, height_max)
    assert 0.0 < mid.item() < 1.0
    # a foot far ABOVE the band clips to the same score as one AT the top of
    # the band -- this is the mechanism the 2026-09-19 world-Z bug exploited
    # (a foot elevated only by standing on a tall terrain patch reads the
    # same as a genuinely high jump)
    at_top = flight_clearance(torch.tensor([[height_max, height_max, height_max, height_max]]), height_min, height_max)
    way_above = flight_clearance(torch.tensor([[5.0, 5.0, 5.0, 5.0]]), height_min, height_max)
    assert torch.allclose(at_top, way_above)


def test_landing_symmetry_is_not_dominated_by_body_length():
    """CONFIRMED regression for the 2026-09-19 landing-symmetry bug: front
    and rear feet touching down at the SAME height must score near-perfect
    symmetry, regardless of how far apart front and rear are horizontally
    (the fixed ~0.38m Go2 hip spacing that dominated the old 3D-distance
    formula). The old formula (norm of 3D centroid difference) would have
    scored this scenario ~exp(-10 * 0.38) = 0.022 no matter what; the fixed
    height-only formula scores it near 1.0."""
    same_height = 0.0
    foot_height = torch.tensor([[same_height, same_height, same_height, same_height]])
    foot_force_z = torch.tensor([[25.0, 25.0, 0.0, 0.0]])  # both front feet in contact

    reward = landing_symmetry(foot_height, foot_force_z, contact_threshold=20.0)

    assert reward.item() > 0.99


def test_landing_symmetry_penalizes_real_height_mismatch():
    foot_height = torch.tensor([[0.10, 0.10, 0.0, 0.0]])  # front feet 0.10m higher than rear
    foot_force_z = torch.tensor([[25.0, 25.0, 0.0, 0.0]])

    reward = landing_symmetry(foot_height, foot_force_z, contact_threshold=20.0)

    assert reward.item() < 0.5  # exp(-10 * 0.10) ~= 0.368


def test_landing_symmetry_gated_on_front_contact():
    foot_height = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    # one front foot not in contact -- should gate to zero even though heights match perfectly
    foot_force_z = torch.tensor([[25.0, 5.0, 0.0, 0.0]])

    reward = landing_symmetry(foot_height, foot_force_z, contact_threshold=20.0)

    assert reward.item() == 0.0
