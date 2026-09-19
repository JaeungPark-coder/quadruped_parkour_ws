"""Pure-tensor CIGR reward math -- deliberately NO isaaclab/omni imports.

CONFIRMED 2026-09-19: `from isaaclab.assets import Articulation` (rewards.py's
own top-level import) pulls in `omni.physics.tensors.impl.api`, which raises
`ModuleNotFoundError: No module named 'omni'` outside a running Isaac
Sim/Kit process -- so nothing in rewards.py, including its module-level
helper functions, could be unit-tested without Isaac Sim actually running.
Splitting the tensor-only math (this file) from the isaaclab-dependent glue
(rewards.py's ManagerTermBase subclasses, which resolve body ids via
ContactSensor.find_bodies and read env.scene) means this file imports with
plain `torch`, and test_cigr_math.py can exercise the actual reward
formulas -- including the two bugs fixed here -- against synthetic foot
tensors, with no Isaac Lab installation required.

Foot ordering is fixed throughout at [FL, FR, RL, RR].
"""
import torch
import torch.nn.functional as F

FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
REAR_IDX = slice(2, 4)   # RL, RR
FRONT_IDX = slice(0, 2)  # FL, FR


def compute_foot_state(foot_pos_w, foot_lin_vel_w, foot_force_history_z, env_origin_z):
    """Derives (N, 4) foot height/planar-speed/contact-force tensors from
    raw per-body simulation tensors.

    foot_pos_w: (N, 4, 3) world-frame foot positions.
    foot_lin_vel_w: (N, 4, 3) world-frame foot linear velocities.
    foot_force_history_z: (N, history, 4) contact-sensor Z force history.
    env_origin_z: (N,) each env's terrain-patch origin height.

    CONFIRMED 2026-09-19, bug fixed here: `foot_pos_w[:, :, 2]` alone is
    world Z, which is meaningless once the owning task inherits rough
    terrain (Isaac-Velocity-Rough-Unitree-Go2) -- each env's terrain patch
    sits at a different origin height, so a robot standing normally on a
    tall sub-terrain read as having permanently elevated feet. That was
    silently saturating `flight_clearance` at its upper clip (constant 1.0,
    zero gradient) and firing `takeoff_height`/`jump_intent` on ordinary
    standing, on terrain-having envs. Subtracting `env_origin_z` corrects
    the per-env CONSTANT offset. It does NOT correct for relief WITHIN one
    env's own terrain patch (a step, ramp or box under one specific foot) --
    that needs a per-foot RayCaster reading the terrain height directly
    under each foot, not added here. Good enough to unblock a flat-terrain
    smoke test and coarse rough-terrain training; revisit before trusting
    numbers on terrain with meaningful per-patch relief (stairs, boxes).
    """
    foot_height = foot_pos_w[:, :, 2] - env_origin_z.unsqueeze(1)
    foot_planar_speed = torch.norm(foot_lin_vel_w[:, :, :2], dim=-1)
    foot_force_z = torch.clamp(torch.max(foot_force_history_z, dim=1)[0], min=0.0)
    return foot_height, foot_planar_speed, foot_force_z


def jump_intent(foot_height, foot_planar_speed, height_threshold, speed_threshold) -> torch.Tensor:
    """I_jump: 1 if >= 2 feet simultaneously clear height_threshold AND
    exceed speed_threshold, else 0."""
    meets_both = (foot_height > height_threshold) & (foot_planar_speed > speed_threshold)
    return (meets_both.sum(dim=1) >= 2).float()


def takeoff_push(foot_force_z, contact_threshold, force_offset, force_scale) -> torch.Tensor:
    """r1_takeoff (Eq. 2): rear feet pushing hard against the ground while
    both are in contact."""
    rear_force = foot_force_z[:, REAR_IDX]
    f_mean = rear_force.mean(dim=1)
    both_in_contact = (rear_force >= contact_threshold).all(dim=1).float()
    return F.softplus((f_mean - force_offset) / force_scale) * both_in_contact


def takeoff_height(foot_height, height_threshold, jump_intent_value) -> torch.Tensor:
    """r2_takeoff (Eq. 3): rear feet lifting above height_threshold, gated
    by jump intent. The "20" scale factor is given explicitly in the
    paper's Eq. (3)."""
    z_hind = foot_height[:, REAR_IDX].mean(dim=1)
    return F.softplus(20.0 * (z_hind - height_threshold)) * jump_intent_value


def takeoff_speed(root_horizontal_speed, max_horizontal_speed, jump_intent_value) -> torch.Tensor:
    """r3_takeoff (Eq. 4): reward sustaining horizontal speed during a
    jump, capped at 1. `root_horizontal_speed` is the root's planar speed
    (norm of body_lin_vel_b's xy components) -- computed in rewards.py,
    which is where the isaaclab Articulation lives."""
    return torch.clamp(root_horizontal_speed / max_horizontal_speed, max=1.0) * jump_intent_value


def flight_clearance(foot_height, height_min, height_max) -> torch.Tensor:
    """r_flight (Eq. 5): all four feet's height clipped to
    [height_min, height_max] and normalized -- keeps swing trajectories in
    a sane band (not clipping obstacles, not wasting energy going too
    high)."""
    clipped = torch.clamp(foot_height, min=height_min, max=height_max)
    return ((clipped - height_min) / (height_max - height_min)).mean(dim=1)


def landing_symmetry(foot_height, foot_force_z, contact_threshold) -> torch.Tensor:
    """r_land (Eq. 6): reward front and rear feet touching down level with
    each other (cat-like symmetric landing), gated by both front feet
    being in contact.

    CONFIRMED 2026-09-19, bug fixed here: the original implementation
    compared front and rear foot CENTROIDS' full 3D positions
    (`norm(front_centroid - rear_centroid)`), which is dominated by the
    fixed front-rear hip spacing (~0.38m on Go2) regardless of gait --
    `exp(-10 * 0.38) ~= 0.022`, and against the paper's own 0.05 weight
    that is a ~0.001 contribution to the total reward in every case, i.e.
    no usable gradient. What little gradient existed pointed at "pull
    front and rear feet together", unrelated to landing symmetry. Reworked
    to compare HEIGHT only -- front and rear touching down at the same
    height -- a quantity the fixed body length cannot dominate.
    """
    front_height = foot_height[:, FRONT_IDX].mean(dim=1)
    rear_height = foot_height[:, REAR_IDX].mean(dim=1)
    height_diff = torch.abs(front_height - rear_height)
    front_force = foot_force_z[:, FRONT_IDX]
    both_front_in_contact = (front_force >= contact_threshold).all(dim=1).float()
    return torch.exp(-10.0 * height_diff) * both_front_in_contact
