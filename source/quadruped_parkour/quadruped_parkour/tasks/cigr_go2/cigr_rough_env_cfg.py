"""Isaac-Velocity-Rough-Unitree-Go2-CIGR-v0: the stock Isaac Lab Go2
rough-terrain velocity task (UnitreeGo2RoughEnvCfg) plus the paper's
Cat-Inspired Gait Reward (CIGR) -- see mdp/rewards.py for the reward itself
and its citation/caveats.

Everything else -- scene, robot, terrain, events, the base rewards/
observations/terminations -- is inherited unchanged from
UnitreeGo2RoughEnvCfg; this file only adds CIGR's five RewTerms (see
mdp/rewards.py's module docstring for why five separate terms rather than
one composite one -- per-component tensorboard visibility).
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
    UnitreeGo2RoughEnvCfg_PLAY,
)

from . import mdp


def _cigr_reward_terms() -> dict[str, RewTerm]:
    asset_cfg = SceneEntityCfg("robot", body_names=[".*_foot"])
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=[".*_foot"])

    return {
        "cigr_takeoff_push": RewTerm(
            func=mdp.TakeoffPushReward,
            weight=0.17,  # paper: R_cat's r1_takeoff coefficient (Eq. 1)
            params={
                "asset_cfg": asset_cfg, "sensor_cfg": sensor_cfg,
                "contact_force_threshold": 20.0,  # paper: 20N
                "takeoff_force_offset": 5.0,      # ADJUST: paper's F_c not recoverable from extracted text
                "takeoff_force_scale": 5.0,       # ADJUST: paper's s_f not recoverable from extracted text
            },
        ),
        "cigr_takeoff_height": RewTerm(
            func=mdp.TakeoffHeightReward,
            weight=0.5,  # paper: R_cat's r2_takeoff coefficient (Eq. 1)
            params={
                "asset_cfg": asset_cfg, "sensor_cfg": sensor_cfg,
                "takeoff_height_threshold": 0.05,  # ADJUST: paper's z_c not recoverable from extracted text
                "jump_height_threshold": 0.1,      # paper: 0.1m
                "jump_speed_threshold": 0.2,       # paper: 0.2 m/s
            },
        ),
        "cigr_takeoff_speed": RewTerm(
            func=mdp.TakeoffSpeedReward,
            weight=1.68,  # paper: R_cat's r3_takeoff coefficient (Eq. 1)
            params={
                "asset_cfg": asset_cfg, "sensor_cfg": sensor_cfg,
                "max_horizontal_speed": 0.8,   # paper: v_max = 0.8 m/s
                "jump_height_threshold": 0.1,  # paper: 0.1m
                "jump_speed_threshold": 0.2,   # paper: 0.2 m/s
            },
        ),
        "cigr_flight_clearance": RewTerm(
            func=mdp.FlightClearanceReward,
            weight=0.01,  # paper: R_cat's r_flight coefficient (Eq. 1)
            params={
                "asset_cfg": asset_cfg, "sensor_cfg": sensor_cfg,
                "flight_height_min": 0.03,  # paper: m_c = 0.03m
                "flight_height_max": 0.12,  # paper: M_c = 0.12m
            },
        ),
        "cigr_landing_symmetry": RewTerm(
            func=mdp.LandingSymmetryReward,
            weight=0.05,  # paper: R_cat's r_land coefficient (Eq. 1)
            params={
                "asset_cfg": asset_cfg, "sensor_cfg": sensor_cfg,
                "contact_force_threshold": 20.0,  # paper: 20N
            },
        ),
    }


@configclass
class UnitreeGo2CigrRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name, term in _cigr_reward_terms().items():
            setattr(self.rewards, name, term)


@configclass
class UnitreeGo2CigrRoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # same CIGR terms at play/eval time so the rendered gait matches what was trained
        for name, term in _cigr_reward_terms().items():
            setattr(self.rewards, name, term)
