"""Cat-Inspired Gait Reward (CIGR), from:

  박상백, 김민표, 이강현, 고병진, 윤종완, 박태준. "고양이 유래 보상 기법을 활용한
  4족 보행 로봇의 익스트림 파쿠르 방법" (Extreme Parkour Method for Quadruped
  Robots Using Cat-Inspired Reward Mechanisms). 2025 한국정보기술학회 추계
  종합학술대회 논문집, pp. 709-712.

R_cat (Eq. 1) = 0.17*r1_takeoff + 0.5*r2_takeoff + 1.68*r3_takeoff
              + 0.01*r_flight + 0.05*r_land

Implemented as a single class-based reward term (ManagerTermBase) rather than
five separate RewTerms, matching how the paper presents R_cat as one composite
-- see CatInspiredGaitReward below for the per-equation breakdown, still kept
as separate private methods for readability/debugging.

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
these gates) or against the paper's full text/appendix if you have it,
rather than trusting the defaults.

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
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# Fixed foot ordering used throughout this file: [FL, FR, RL, RR].
_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
_REAR_IDX = slice(2, 4)   # RL, RR
_FRONT_IDX = slice(0, 2)  # FL, FR


class CatInspiredGaitReward(ManagerTermBase):
    """R_cat (Eq. 1): jump-takeoff push/height/speed shaping + flight foot
    clearance + landing symmetry, modeled on cat gait biomechanics.

    Resolves FL_foot/FR_foot/RL_foot/RR_foot to fixed body indices once in
    __init__ via ContactSensor.find_bodies -- the same pattern
    isaaclab_tasks' Spot GaitReward uses -- rather than relying on a
    wildcard SceneEntityCfg match to come back in a particular order.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]

        self.foot_body_ids = [self.contact_sensor.find_bodies([name])[0][0] for name in _FOOT_NAMES]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        contact_force_threshold: float = 20.0,
        takeoff_force_offset: float = 5.0,
        takeoff_force_scale: float = 5.0,
        takeoff_height_threshold: float = 0.05,
        jump_height_threshold: float = 0.1,
        jump_speed_threshold: float = 0.2,
        max_horizontal_speed: float = 0.8,
        flight_height_min: float = 0.03,
        flight_height_max: float = 0.12,
    ) -> torch.Tensor:
        # (N, 4, ...) tensors, foot order [FL, FR, RL, RR] per self.foot_body_ids
        foot_pos = self.asset.data.body_pos_w[:, self.foot_body_ids, :]
        foot_height = foot_pos[:, :, 2]
        foot_planar_speed = torch.norm(self.asset.data.body_lin_vel_w[:, self.foot_body_ids, :2], dim=-1)
        # max over the contact sensor's history window -- robust to a single
        # missed substep, same pattern isaaclab_tasks' foot_slip_penalty uses.
        foot_force_z = torch.clamp(
            torch.max(self.contact_sensor.data.net_forces_w_history[:, :, self.foot_body_ids, 2], dim=1)[0],
            min=0.0,
        )

        jump_intent = self._jump_intent(foot_height, foot_planar_speed, jump_height_threshold, jump_speed_threshold)

        r1 = self._takeoff_push(foot_force_z, contact_force_threshold, takeoff_force_offset, takeoff_force_scale)
        r2 = self._takeoff_height(foot_height, takeoff_height_threshold, jump_intent)
        r3 = self._takeoff_speed(self.asset, max_horizontal_speed, jump_intent)
        r_flight = self._flight_clearance(foot_height, flight_height_min, flight_height_max)
        r_land = self._landing_symmetry(foot_pos, foot_force_z, contact_force_threshold)

        return 0.17 * r1 + 0.5 * r2 + 1.68 * r3 + 0.01 * r_flight + 0.05 * r_land

    """
    Per-equation helpers (Eq. 2-6 of the paper).
    """

    @staticmethod
    def _jump_intent(foot_height, foot_planar_speed, height_threshold, speed_threshold) -> torch.Tensor:
        """I_jump: 1 if >= 2 feet simultaneously clear height_threshold AND
        exceed speed_threshold, else 0."""
        meets_both = (foot_height > height_threshold) & (foot_planar_speed > speed_threshold)
        return (meets_both.sum(dim=1) >= 2).float()

    @staticmethod
    def _takeoff_push(foot_force_z, contact_threshold, force_offset, force_scale) -> torch.Tensor:
        """r1_takeoff (Eq. 2): rear feet pushing hard against the ground
        while both are in contact."""
        rear_force = foot_force_z[:, _REAR_IDX]
        f_mean = rear_force.mean(dim=1)
        both_in_contact = (rear_force >= contact_threshold).all(dim=1).float()
        return F.softplus((f_mean - force_offset) / force_scale) * both_in_contact

    @staticmethod
    def _takeoff_height(foot_height, height_threshold, jump_intent) -> torch.Tensor:
        """r2_takeoff (Eq. 3): rear feet lifting above height_threshold,
        gated by jump intent. The "20" scale factor is given explicitly in
        the paper's Eq. (3)."""
        z_hind = foot_height[:, _REAR_IDX].mean(dim=1)
        return F.softplus(20.0 * (z_hind - height_threshold)) * jump_intent

    @staticmethod
    def _takeoff_speed(asset: Articulation, max_horizontal_speed, jump_intent) -> torch.Tensor:
        """r3_takeoff (Eq. 4): reward sustaining horizontal speed during a
        jump, capped at 1."""
        v_xy = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
        return torch.clamp(v_xy / max_horizontal_speed, max=1.0) * jump_intent

    @staticmethod
    def _flight_clearance(foot_height, height_min, height_max) -> torch.Tensor:
        """r_flight (Eq. 5): all four feet's height clipped to
        [height_min, height_max] and normalized -- keeps swing trajectories
        in a sane band (not clipping obstacles, not wasting energy going too
        high)."""
        clipped = torch.clamp(foot_height, min=height_min, max=height_max)
        return ((clipped - height_min) / (height_max - height_min)).mean(dim=1)

    @staticmethod
    def _landing_symmetry(foot_pos, foot_force_z, contact_threshold) -> torch.Tensor:
        """r_land (Eq. 6): reward a small offset between the front-feet and
        rear-feet centroids at touchdown (cat-like symmetric landing),
        gated by both front feet being in contact."""
        front_centroid = foot_pos[:, _FRONT_IDX, :].mean(dim=1)
        rear_centroid = foot_pos[:, _REAR_IDX, :].mean(dim=1)
        d_foot = torch.norm(front_centroid - rear_centroid, dim=-1)
        front_force = foot_force_z[:, _FRONT_IDX]
        both_front_in_contact = (front_force >= contact_threshold).all(dim=1).float()
        return torch.exp(-10.0 * d_foot) * both_front_in_contact
