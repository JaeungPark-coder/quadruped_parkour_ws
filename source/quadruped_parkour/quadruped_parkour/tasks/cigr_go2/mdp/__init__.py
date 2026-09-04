"""MDP terms for the CIGR Go2 task: the stock Isaac Lab locomotion-velocity
terms (track_lin_vel_xy_exp, feet_air_time, dof_torques_l2, ...) plus this
task's own Cat-Inspired Gait Reward -- same re-export convention
isaaclab_tasks' own robot-specific mdp packages (e.g. config/spot/mdp) use.
"""
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401,F403

from .rewards import *  # noqa: F401,F403
