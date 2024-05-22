from pathlib import Path
from typing import Sequence
import gymnasium as gym
import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf, open_dict
from tianshou.utils.net.common import Net
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

from typing import Dict, List, Optional, Union, Sequence, Any, Tuple, Callable, TypeVar, Generic, cast, Type, Mapping
from torch import nn
import torch
import numpy as np
from tianshou.utils.net.common import MLP
from utils.delay import DelayedRoboticEnv, StickyActionWrapper, GaussianNoiseActionWrapper, GaussianNoiseObservationWrapper, NormedObsActWrapper, MaxStepWrapper, BetterResetWrapper, MergeObsActWrapper

ModuleType = Type[nn.Module]


import logging

from pytorch_lightning.utilities import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_pylogger(__name__)


from pytorch_lightning.utilities.logger import (
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
)

def make_env(env_cfg):
    if env_cfg.name.startswith("gym_anm"): # some env does not support gymnasium
        import gym
    else:
        import gymnasium as gym
    
    if env_cfg.use_contact_forces:
        env = gym.make(env_cfg.name, use_contact_forces=True)
    else:
        if env_cfg.name.startswith("Panda"): 
            import panda_gym
        env = gym.make(env_cfg.name)

    if env_cfg.name.startswith("Panda"):
        env = MergeObsActWrapper(env)
        env = PandaAddStateWrapper(env)
        if "Push" in env_cfg.name:
            env = PusherRewardWrapper(env, env_cfg)
    
    if env_cfg.gaussian_obs: # ! order is important. when operating on obs, this wrapper must be inside
        env = GaussianNoiseObservationWrapper(env, env_cfg.gaussian_obs)
    
    env = DelayedRoboticEnv(env, env_cfg.delay, env_cfg.fixed_delay, env_cfg.global_cfg)

    if env_cfg.name.startswith("gym_anm"):
        env = NormedObsActWrapper(env)
        env = MaxStepWrapper(env, env_cfg.max_step)

    if env_cfg.sticky_action_prob:
        env = StickyActionWrapper(env, env_cfg.sticky_action_prob)
    if env_cfg.noise_fraction:
        env = GaussianNoiseActionWrapper(env, env_cfg.noise_fraction)

    env = BetterResetWrapper(env) # avoid reset(**kwargs) error

    if env_cfg.save_minari: # to save dataset
        # from minari import DataCollector
        from minari import DataCollector
        env = DataCollector(env, record_infos=True, observation_space=env.observation_space, action_space=env.action_space) 

    return env

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def config_format(cfg: DictConfig) -> DictConfig:
    """Formats config to be saved to wandb."""
    params = _convert_params(_flatten_dict(_sanitize_callable_params(cfg)))
    return params

def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "task_name", 
        "tags", 
        "env", 
        "net",
        "policy", 
        "optimizer", 
        "train_collector", 
        "test_collector",
        "trainer",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)

class RNNNet(nn.Module):
    """ use same parameter with Net, but add a RNN layer before the MLP
    extra parameters:
        rnn_size: int, default 256 # add a RNN layer before the MLP
    """
    def __init__(self, state_shape, **kwargs):
        super().__init__()
        # 
        kwargs["state_shape"] = state_shape
        if "concat" in kwargs and kwargs["concat"]: # use as critic or actor
            input_dim = kwargs["state_shape"][0] + kwargs["action_shape"][0]
        else:
            input_dim = kwargs["state_shape"][0]
        # rnn
        self.rnn = None # TODO
        rnn_output_dim = rnn_state_dim = kwargs.pop("rnn_size", 256)
        self.rnn = nn.RNN(input_dim, rnn_state_dim, batch_first=True)
        # mlp
        kwargs["state_shape"] = (rnn_output_dim,)
        # self.net = Net(**kwargs) 
        self.net = MLP(rnn_output_dim, kwargs["hidden_sizes"][-1], kwargs["hidden_sizes"][:-1])
        # 
        self.output_dim = kwargs["hidden_sizes"][-1]
        self.device = kwargs["device"]
        self.softmax = kwargs["softmax"]
        self.to(self.device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Any]:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        x, hidden_s = self.rnn(obs)
        logits = self.net(x)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

class RNNNetX(nn.Module):
    """ A simple RNN network behaving like tianshou.utils.net.common.Net
    """
    def __init__(self, state_shape, action_shape=0, num_atoms=1, **kwargs):
        super().__init__()
        self.softmax = kwargs.pop('softmax', False)
        self.device = kwargs.pop('device', 'cpu')
        hidden_sizes = kwargs.pop('hidden_sizes', [256, 256])
        print(kwargs)
        self.num_atoms = num_atoms
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        self.output_dim = action_dim
        self.output_dim = self.output_dim or hidden_sizes[-1]
        # build pytorch RNN network
        self.rnn = nn.RNN(input_dim, hidden_sizes[0], batch_first=True)
        self.net = MLP(hidden_sizes[0], action_dim, hidden_sizes[1:])
        self.model = nn.Sequential(self.rnn, self.net)
        self.to(self.device)
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Any]:
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        x, hidden_s = self.rnn(obs)
        logits = self.net(x)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

class DummyNumEnv(HalfCheetahEnv):
    """A dummy environment that generate obs number arabic number from 0 to inf.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.max_step = 100

    def reset(self):
        """ rewrite obs to numbers
        """
        self.count = 0
        res = super().reset()
        if isinstance(res, tuple):
            obs = res[0]
            obs = np.ones_like(obs) * self.count * 0.01
            res = (obs, *res[1:])
        return res

    def step(self, action):
        """ rewrite obs to numbers
        """
        self.count += 1
        res = super().step(action)
        obs = res[0]
        obs = (np.ones_like(obs) * self.count)
        res = (obs, *res[1:])
        if self.count > 100:
            # res[2] = True
            res = (obs, res[1], True, *res[3:])
        # reward: 1 - (action - self.count) / 100
        reward = 1 - (action[0] - self.count) / 100
        res = (obs, reward, *res[2:])
        return res


from gym.envs.registration import register
register(
    id='DummyNum-v0',
    entry_point='utils.__init__:DummyNumEnv',
)

import gymnasium as gym
gym.envs.registration.register(
    id='DummyNum-v0',
    entry_point='utils.__init__:DummyNumEnv',
)


def idx_stack(indices, buffer, step, direction="prev"):
    """
    input: 
        indices: [idx_0, idx_1, ..., idx_n] # shape: (batch_size,)
        buffer: ReplayBuffer
        step: int
        direction: str
    output:
        indices_res: shape: (batch_size, step)
        if direction == "prev":
            [
                [idx_0-T+1, idx_0-T+2, ..., idx_0],
                ...
            ]
        if direction == "next":
            [
                [idx_0, idx_0+1, ..., idx_0+T-1],
                ...
            ]
        ps. would keep the same if meet the next episode
    """
    assert direction in ["prev", "next"], "direction must be prev or next"
    indices_buf = []
    last_indices = indices
    for _ in range(step):
        indices_buf.insert(0, last_indices) if direction == "prev" else indices_buf.append(last_indices)
        last_indices = buffer.prev(last_indices) if direction == "prev" else buffer.next(last_indices)
    indices_res = np.stack(indices_buf, axis=-1)
    return indices_res

def idx_prev_stack(indices, buffer, num):
    return idx_stack(indices, buffer, num, direction="prev")

def idx_next_stack(indices, buffer, num):
    return idx_stack(indices, buffer, num, direction="next")

def idx_is_start(indices, buffer):
    return buffer.prev(indices) == indices

def idx_is_end(indices, buffer):
    return buffer.next(indices) == indices



def move_all_files(src_dir: str, dst_dir: str) -> None:
    import os
    import shutil

    os.makedirs(dst_dir, exist_ok=True)

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        shutil.move(src_path, dst_path)

def copy_all_files(src_dir: str, dst_dir: str) -> None:
    import os
    import shutil
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)