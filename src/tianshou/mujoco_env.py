import warnings

import gymnasium as gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs

TURN_ON_ENVPOOL = False
if TURN_ON_ENVPOOL:
    try:
        import envpool
    except ImportError:
        envpool = None
else:
    envpool = None


from gymnasium.envs.registration import register

import numpy as np
from queue import Queue
from copy import deepcopy
import utils


# HalfCheetah-v4-delay_0

class Config:
    pass


class Globa_cfg:
    def __init__(self):
        self.actor_input = Config()
        self.critic_input = Config()
        self.actor_input.history_merge_method = "none"
        self.critic_input.history_merge_method = "none"
        self.history_num = 0


for base_env_name in ["HalfCheetah-v4", "Ant-v4", "Hopper-v4", "Walker2d-v4", "Humanoid-v4"]:
    for delay in range(16):
        register(
            f"{base_env_name}-delay_{delay}",
            entry_point="src.tianshou.mujoco_env:DelayedEnvWrapper",
            kwargs=dict(
                base_env=gym.make(base_env_name),
                delay_steps=delay,
                fixed_delay=True,
                global_config=Globa_cfg(),
            ),
            max_episode_steps=5000,
        )

class DelayedEnvWrapper(utils.delay.DelayedRoboticEnv):
    metadata = {'render.modes': ['human', 'text']}
    def __init__(self, base_env: gym.Env, delay_steps, fixed_delay, global_config=None):
        super().__init__(base_env, delay_steps, fixed_delay, global_config)

# class DelayedEnvWrapper(gym.Wrapper):
#     metadata = {'render.modes': ['human', 'text']}
#     def __init__(self, base_env: gym.Env, delay_steps, fixed_delay, global_config=None):
#         # For debugging: replace 'base_env_name' with 'HalfCheetah-v4'
#         super().__init__(base_env)
        

def make_mujoco_env(task, seed, training_num, test_num, obs_norm):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(
            task, num_envs=training_num, seed=seed
        )
        test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Mujoco environments more efficiently."
        )
        env = gym.make(task)
        train_envs = ShmemVectorEnv(
            [lambda: gym.make(task) for _ in range(training_num)]
        )
        test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
        if TURN_ON_ENVPOOL:
            env.seed(seed)
            train_envs.seed(seed)
            test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs