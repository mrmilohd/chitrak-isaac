import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_flat_env_cfg:ChitrakFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Chitrak-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chitrak_flat_env_cfg:ChitrakFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChitrakFlatPPORunnerCfg",
    },
)
