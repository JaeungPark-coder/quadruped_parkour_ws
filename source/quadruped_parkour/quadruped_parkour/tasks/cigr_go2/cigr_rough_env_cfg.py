"""Isaac-Velocity-Rough-Unitree-Go2-CIGR-v0: the stock Isaac Lab Go2
rough-terrain velocity task (UnitreeGo2RoughEnvCfg) plus the paper's
Cat-Inspired Gait Reward (CIGR) -- see mdp/rewards.py for the reward itself
and its citation/caveats.

Everything else -- scene, robot, terrain, events, the base rewards/
observations/terminations -- is inherited unchanged from
UnitreeGo2RoughEnvCfg; this file only adds one extra RewTerm.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
    UnitreeGo2RoughEnvCfg_PLAY,
)

from . import mdp


def _cat_inspired_gait_term() -> RewTerm:
    return RewTerm(
        func=mdp.CatInspiredGaitReward,
        # R_cat's five component weights (0.17/0.5/1.68/0.01/0.05, Eq. 1) are
        # already applied inside CatInspiredGaitReward itself -- this is the
        # overall weight on the composite term.
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "contact_force_threshold": 20.0,      # paper: 20N
            "takeoff_force_offset": 5.0,          # ADJUST: paper's F_c not recoverable from extracted text
            "takeoff_force_scale": 5.0,           # ADJUST: paper's s_f not recoverable from extracted text
            "takeoff_height_threshold": 0.05,     # ADJUST: paper's z_c not recoverable from extracted text
            "jump_height_threshold": 0.1,         # paper: 0.1m
            "jump_speed_threshold": 0.2,          # paper: 0.2 m/s
            "max_horizontal_speed": 0.8,          # paper: v_max = 0.8 m/s
            "flight_height_min": 0.03,            # paper: m_c = 0.03m
            "flight_height_max": 0.12,            # paper: M_c = 0.12m
        },
    )


@configclass
class UnitreeGo2CigrRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.cat_inspired_gait = _cat_inspired_gait_term()


@configclass
class UnitreeGo2CigrRoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # same CIGR term at play/eval time so the rendered gait matches what was trained
        self.rewards.cat_inspired_gait = _cat_inspired_gait_term()
