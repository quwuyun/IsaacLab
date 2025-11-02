# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
AMP ExoHumanoid locomotion environment.
"""

import gymnasium as gym

from . import agents

# from .exo_humanoid_amp_env import ExoHumanoidAmpEnv
# from .exo_humanoid_amp_env_cfg import ExoHumanoidAmpDanceEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Exo-Humanoid-AMP-Distillation-Dance-Direct-v0",
    entry_point=f"{__name__}.exo_humanoid_amp_distillation_env:ExoHumanoidAmpDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exo_humanoid_amp_distillation_env_cfg:ExoHumanoidAmpDistillationDanceEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_dance_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Exo-Humanoid-AMP-Distillation-Run-Direct-v0",
    entry_point=f"{__name__}.exo_humanoid_amp_distillation_env:ExoHumanoidAmpDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exo_humanoid_amp_distillation_env_cfg:ExoHumanoidAmpDistillationRunEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_run_amp_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Exo-Humanoid-AMP-Distillation-Walk-Direct-v0",
    entry_point=f"{__name__}.exo_humanoid_amp_distillation_env:ExoHumanoidAmpDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.exo_humanoid_amp_distillation_env_cfg:ExoHumanoidAmpDistillationWalkEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_walk_amp_cfg.yaml",
    },
)
