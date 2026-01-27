# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP Humanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Knee-Effort-Rugged-Dance-Direct-v0",
    entry_point=f"{__name__}.knee_effort_rugged_env:KneeEffortRuggedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_rugged_env_cfg:KneeEffortRuggedDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Knee-Effort-Rugged-Run-Direct-v0",
    entry_point=f"{__name__}.knee_effort_rugged_env:KneeEffortRuggedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_rugged_env_cfg:KneeEffortRuggedRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Knee-Effort-Rugged-Walk-Direct-v0",
    entry_point=f"{__name__}.knee_effort_rugged_env:KneeEffortRuggedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_rugged_env_cfg:KneeEffortRuggedWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)
