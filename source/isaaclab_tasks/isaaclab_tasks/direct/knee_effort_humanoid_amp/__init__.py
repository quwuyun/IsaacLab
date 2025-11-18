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
    id="Isaac-Knee-Effort-Humanoid-AMP-Dance-Direct-v0",
    entry_point=f"{__name__}.knee_effort_humanoid_amp_env:KneeEffortHumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_humanoid_amp_env_cfg:KneeEffortHumanoidAmpDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Knee-Effort-Humanoid-AMP-Run-Direct-v0",
    entry_point=f"{__name__}.knee_effort_humanoid_amp_env:KneeEffortHumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_humanoid_amp_env_cfg:KneeEffortHumanoidAmpRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Knee-Effort-Humanoid-AMP-Walk-Direct-v0",
    entry_point=f"{__name__}.knee_effort_humanoid_amp_env:KneeEffortHumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_humanoid_amp_env_cfg:KneeEffortHumanoidAmpWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)
