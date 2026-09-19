"""Cat-Inspired Gait Reward (CIGR), from:

  박상백, 김민표, 이강현, 고병진, 윤종완, 박태준. "고양이 유래 보상 기법을 활용한
  4족 보행 로봇의 익스트림 파쿠르 방법" (Extreme Parkour Method for Quadruped
  Robots Using Cat-Inspired Reward Mechanisms). 2025 한국정보기술학회 추계
  종합학술대회 논문집, pp. 709-712.

R_cat (Eq. 1) = 0.17*r1_takeoff + 0.5*r2_takeoff + 1.68*r3_takeoff
              + 0.01*r_flight + 0.05*r_land

Implemented as FIVE separate RewTerms (one per paper equation), each with its
own weight matching the paper's coefficients, rather than one composite
ManagerTermBase class that sums them internally with weight=1.0. The total
reward Isaac Lab's RewardManager arrives at is mathematically identical
either way (it sums every configured term's weight * value regardless of how
many terms there are) -- the only difference is that Isaac Lab's existing
per-RewTerm logging then shows each of r1_takeoff/r2_takeoff/r3_takeoff/
r_flight/r_land separately in tensorboard (Episode_Reward/cigr_*), instead of
only their pre-summed total. That per-component visibility is the whole
point of this split: the five weights span a 168x range (0.01 to 1.68), a
common recipe for one term silently dominating training (reward hacking) --
being able to see whether e.g. r3_takeoff (0.68 weight) alone is driving the
score, or whether r_flight (0.01 weight) is actually contributing anything
at all, is what you need to diagnose that. Splitting the class means
foot_pos/foot_height/foot_force_z/jump_intent get recomputed per term
instead of shared once -- a small duplicated-compute cost against 4 feet's
worth of tensors, traded for that diagnostic visibility.

The actual reward formulas live in cigr_math.py, not here -- see that
file's own docstring for why (importing isaaclab.assets pulls in `omni`,
which does not exist outside a running Isaac Sim process, so nothing in
this file can be unit-tested without Isaac Sim actually running; the
formulas themselves need only torch and are tested against synthetic
tensors in test_cigr_math.py). This file is the isaaclab-dependent glue:
resolving FL/FR/RL/RR to fixed body indices via ContactSensor.find_bodies,
and reading the real per-step simulation tensors those formulas consume.

CONFIRMED 2026-09-19, two bugs found reading this file (not yet exercised
against a live run) and fixed in cigr_math.py:

1. Foot height was plain world Z (`foot_pos_w[:, :, 2]`), meaningless once
   this task inherits rough terrain (Isaac-Velocity-Rough-Unitree-Go2) --
   each env's terrain patch sits at a different origin height, so standing
   normally on a tall sub-terrain read as permanently elevated feet. This
   was silently saturating r_flight at its upper clip (constant 1.0, zero
   gradient) and firing r2_takeoff/jump_intent on ordinary standing.
   `cigr_math.compute_foot_state` now subtracts each env's terrain-patch
   origin height -- corrects the per-env constant offset, not relief
   within one env's own patch (a step or ramp under one specific foot),
   which needs a per-foot RayCaster not added here.
2. r_land compared front/rear foot CENTROIDS' full 3D positions, dominated
   by the ~0.38m front-rear hip spacing on Go2 regardless of gait --
   `exp(-10 * 0.38) ~= 0.022`, effectively a constant with no usable
   gradient (and what little gradient existed pointed at "pull the feet
   together", unrelated to landing symmetry). `cigr_math.landing_symmetry`
   now compares front/rear HEIGHT only, which the fixed body length can't
   dominate.

ADJUST: the paper's F_c (Eq. 2, a force offset inside the softplus) and z_c
(Eq. 3, a rear-foot-height threshold) are named in the paper's text but their
numeric values were not present in the extracted PDF content (most likely
lost in a table/formula the extraction missed -- everything else here, the
20N contact threshold, the 0.1m/0.2m/s jump-intent thresholds, v_max=0.8m/s,
m_c=0.03m, M_c=0.12m, and all five R_cat weights, IS given explicitly and is
used exactly as stated). takeoff_force_offset/takeoff_force_scale and
takeoff_height_threshold below are reasonable placeholders standing in for
F_c/s_f and z_c -- tune them against training behavior (watch whether the
robot starts taking unnecessary hops, per the paper's own stated purpose for
these gates), ideally now WITH each component's own tensorboard curve to see
which one actually needs it, or against the paper's full text/appendix if
you have it, rather than trusting the defaults.

Written and reasoned about against the Isaac Lab APIs confirmed live from
isaac-sim/IsaacLab's actual source (isaaclab_tasks...velocity.mdp.rewards and
...config.spot.mdp.rewards, particularly the GaitReward class's use of
ContactSensor.find_bodies for per-body indexing), but WITHOUT the ability to
run Isaac Lab in the environment this was authored in -- treat this as a
solid, carefully-grounded first draft, not verified to run. Smoke-test with
a short --num_envs 64 training run before trusting it for a full run.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .cigr_math import (
    FOOT_NAMES,
    compute_foot_state,
    flight_clearance,
    jump_intent,
    landing_symmetry,
    takeoff_height,
    takeoff_push,
    takeoff_speed,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


class _FootIndexedRewardTerm(ManagerTermBase):
    """Shared per-body-id resolution for all five CIGR components. Resolves
    FL_foot/FR_foot/RL_foot/RR_foot to fixed body indices once in __init__
    via ContactSensor.find_bodies -- the same pattern isaaclab_tasks' Spot
    GaitReward uses -- rather than relying on a wildcard SceneEntityCfg
    match to come back in a particular order."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.foot_body_ids = [self.contact_sensor.find_bodies([name])[0][0] for name in FOOT_NAMES]

    def _foot_state(self):
        """(N, 4, ...) tensors, foot order [FL, FR, RL, RR] per self.foot_body_ids.

        See cigr_math.compute_foot_state's own docstring for why
        env_origins is subtracted from raw world-frame foot height."""
        foot_pos = self.asset.data.body_pos_w[:, self.foot_body_ids, :]
        foot_lin_vel = self.asset.data.body_lin_vel_w[:, self.foot_body_ids, :]
        foot_force_history_z = self.contact_sensor.data.net_forces_w_history[:, :, self.foot_body_ids, 2]
        env_origin_z = self._env.scene.env_origins[:, 2]
        foot_height, foot_planar_speed, foot_force_z = compute_foot_state(
            foot_pos, foot_lin_vel, foot_force_history_z, env_origin_z)
        return foot_pos, foot_height, foot_planar_speed, foot_force_z


class TakeoffPushReward(_FootIndexedRewardTerm):
    """r1_takeoff (Eq. 2). Paper weight: 0.17."""

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 contact_force_threshold: float = 20.0, takeoff_force_offset: float = 5.0,
                 takeoff_force_scale: float = 5.0) -> torch.Tensor:
        _, _, _, foot_force_z = self._foot_state()
        return takeoff_push(foot_force_z, contact_force_threshold, takeoff_force_offset, takeoff_force_scale)


class TakeoffHeightReward(_FootIndexedRewardTerm):
    """r2_takeoff (Eq. 3). Paper weight: 0.5."""

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 takeoff_height_threshold: float = 0.05, jump_height_threshold: float = 0.1,
                 jump_speed_threshold: float = 0.2) -> torch.Tensor:
        _, foot_height, foot_planar_speed, _ = self._foot_state()
        intent = jump_intent(foot_height, foot_planar_speed, jump_height_threshold, jump_speed_threshold)
        return takeoff_height(foot_height, takeoff_height_threshold, intent)


class TakeoffSpeedReward(_FootIndexedRewardTerm):
    """r3_takeoff (Eq. 4). Paper weight: 1.68."""

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 max_horizontal_speed: float = 0.8, jump_height_threshold: float = 0.1,
                 jump_speed_threshold: float = 0.2) -> torch.Tensor:
        _, foot_height, foot_planar_speed, _ = self._foot_state()
        intent = jump_intent(foot_height, foot_planar_speed, jump_height_threshold, jump_speed_threshold)
        v_xy = torch.norm(self.asset.data.root_lin_vel_b[:, :2], dim=-1)
        return takeoff_speed(v_xy, max_horizontal_speed, intent)


class FlightClearanceReward(_FootIndexedRewardTerm):
    """r_flight (Eq. 5). Paper weight: 0.01."""

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 flight_height_min: float = 0.03, flight_height_max: float = 0.12) -> torch.Tensor:
        _, foot_height, _, _ = self._foot_state()
        return flight_clearance(foot_height, flight_height_min, flight_height_max)


class LandingSymmetryReward(_FootIndexedRewardTerm):
    """r_land (Eq. 6). Paper weight: 0.05."""

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
                 contact_force_threshold: float = 20.0) -> torch.Tensor:
        _, foot_height, _, foot_force_z = self._foot_state()
        return landing_symmetry(foot_height, foot_force_z, contact_force_threshold)
