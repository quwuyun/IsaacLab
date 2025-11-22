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
    id="Isaac-Knee-Effort-Humanoid-Distillation-Walk-Direct-v0",
    entry_point=f"{__name__}.knee_effort_humanoid_distillation_env:KneeEffortHumanoidDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.knee_effort_humanoid_distillation_env_cfg:KneeEffortHumanoidDistillationEnvCfg",
        # "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KneeEffortHumanoidPPORunnerCfg",
    },
)
