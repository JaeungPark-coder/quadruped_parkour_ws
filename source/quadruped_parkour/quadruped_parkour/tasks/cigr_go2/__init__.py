"""Registers the CIGR Go2 task. Reuses the stock Go2 rough-terrain PPO
runner config (isaaclab_tasks...config.go2.agents.rsl_rl_ppo_cfg) unchanged
-- CIGR only changes the reward function, not the PPO hyperparameters.
"""
import gymnasium as gym

from . import cigr_rough_env_cfg  # noqa: F401 -- imported for its side effect free module load; referenced by string entry points below

_GO2_AGENTS_MODULE = "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents"

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-CIGR-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cigr_rough_env_cfg:UnitreeGo2CigrRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_GO2_AGENTS_MODULE}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-CIGR-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cigr_rough_env_cfg:UnitreeGo2CigrRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_GO2_AGENTS_MODULE}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)
