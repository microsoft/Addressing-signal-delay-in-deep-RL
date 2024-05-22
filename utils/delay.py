import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from queue import Queue
from copy import deepcopy
from gymnasium.utils import RecordConstructorArgs

# main
class DelayedRoboticEnv(gym.Wrapper, RecordConstructorArgs):
    """
    Args:
        fixed_delay: if True, the delay_steps is fixed. Otherwise, 
            the delay_steps is sampled from a uniform distribution
            between [0, max_delay_steps)
            self.delay_buf = [delay=max, delay=max-1, ..., delay=0]
            idx = - (delay + 1) + max
        delay_keep_order_method:
            "none": random sample from the delay_buf
            "expect1": sample from the delay_buf with the step forward of 1 [0,1,2]
    Returns:
        info:
            obs_next_nodelay: the next observation without delay
            obs_next_delayed: the next observation with delay
            historical_act_cur: the historical actions of the next step [a_{t-max-1}, ..., a_{t-2}]
            historical_act_next: the historical actions of the next step [a_{t-max}, ..., a_{t-1}]
            historical_obs_cur: the historical observations of the next step [o_{t-max}, ..., o_{t-1}]
            historical_obs_next: the historical observations of the next step [o_{t-max+1}, ..., o_{t}]

    """
    metadata = {'render.modes': ['human', 'text']}

    def __init__(
            self, 
            base_env: gym.Env, 
            delay_steps, 
            fixed_delay, 
            global_config
        ):
        super().__init__(base_env)
        
        self.env = base_env
        self.delay_steps = delay_steps
        self.fixed_delay = fixed_delay
        self.global_cfg = global_config

        # setup for delayed observations
        self.delay_buf = ListAsQueue(maxsize=delay_steps+1)
        self.last_oracle_obs = None
        self.last_delayed_step = None # for debug

        # setup history merge
        if self.global_cfg.history_num:
            self.history_num = self.global_cfg.history_num
        else:
            self.history_num = 0

    def reset(self):
        # pre: adapt to different envs
        res = self.env.reset()
        if isinstance(res, tuple): obs_next_nodelay, info = res
        else: obs_next_nodelay, info = res, {}
        
        # reset delay_buf - empty then fill the delay_buf with zeros
        while not self.delay_buf.empty(): self.delay_buf.get()
        while not self.delay_buf.full(): self.delay_buf.put(np.zeros_like(obs_next_nodelay))
        
        # reset act_buf,prev_act - empty then fill the act_buf with zeros
        if self.history_num > 0:
            self.act_buf = [np.zeros(self.env.action_space.shape) for _ in range(self.history_num)]
            self.obs_buf = [np.zeros_like(obs_next_nodelay) for _ in range(self.history_num)]
        else:
            pass
        
        # update delay_buf
        self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay) # [max,max-1, ..., 1, 0]

        # get index
        if not self.fixed_delay:
            if not self.global_cfg.debug.delay_keep_order_method:
                self.last_delayed_step = np.random.randint(0, self.delay_steps+1) if self.delay_steps > 0 else 0
            elif self.global_cfg.debug.delay_keep_order_method == "expect1":
                self.last_delayed_step = self.delay_steps # start from the max delay step
                # self.last_delayed_step = self.delay_steps // 2 # start from the middle delay step
                self.last_delayed_step = np.random.randint(self.last_delayed_step-1, self.last_delayed_step+2)
                self.last_delayed_step = np.clip(self.last_delayed_step, 0, self.delay_steps)
            else:
                raise ValueError("Invalid delay_keep_order_method {}".format(self.global_cfg.debug.delay_keep_order_method))
        else:
            self.last_delayed_step = self.delay_steps

        # get
        obs_next_delayed = self.delay_buf[self.delay_steps - self.last_delayed_step] # 0 -> 0

        # info
        if self.history_num > 0:
            self.obs_buf.append(obs_next_delayed)
            self.obs_buf.pop(0)
            info["historical_act_next"] = np.stack(self.act_buf, axis=0)
            info["historical_act_cur"] = np.stack(self.act_buf, axis=0)
            info["historical_obs_next"] = np.stack(self.obs_buf, axis=0)
            info["historical_obs_cur"] = np.stack(self.obs_buf, axis=0)
        else:
            info["historical_act_next"] = False
            info["historical_act_cur"] = False
            info["historical_obs_next"] = False
            info["historical_obs_cur"] = False


        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        info["obs_nodelay"] = None
        info["obs_delayed_step_num"] = self.last_delayed_step
        
        # end
        self.last_oracle_obs = obs_next_nodelay

        return obs_next_delayed, info

    def preprocess_fn(self, res, action):
        """
        preprocess the observation before the agent decision
        """
        # pre: adapt to different envs
        if len(res) == 4: 
            obs_next_nodelay, reward, done, info = res
            truncated = False
        elif len(res) == 5:
            obs_next_nodelay, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        
        # update delay_buf
        obs_next_delayed = self.delay_buf.get()
        self.delay_buf.put(obs_next_nodelay)

        # get index
        if not self.fixed_delay: # replace obs_next_delayed and self.last_delayed_step
            if not self.global_cfg.debug.delay_keep_order_method:
                self.last_delayed_step = np.random.randint(0, self.delay_steps+1) if self.delay_steps > 0 else 0
            elif self.global_cfg.debug.delay_keep_order_method == "expect1":
                self.last_delayed_step = np.random.randint(self.last_delayed_step-1, self.last_delayed_step+2)
                self.last_delayed_step = np.clip(self.last_delayed_step, 0, self.delay_steps)
            else:
                raise ValueError("Invalid delay_keep_order_method {}".format(self.global_cfg.debug.delay_keep_order_method))
        else:
            self.last_delayed_step = self.delay_steps
        
        # get
        obs_next_delayed = self.delay_buf[self.delay_steps - self.last_delayed_step]
        
        
        info["obs_next_nodelay"] = obs_next_nodelay
        info["obs_next_delayed"] = obs_next_delayed
        info["obs_nodelay"] = self.last_oracle_obs
        info["obs_delayed_step_num"] = self.last_delayed_step
        
        # end
        self.last_oracle_obs = obs_next_nodelay

        # act merge
        if self.history_num > 0:
            info["historical_act_cur"] = np.stack(self.act_buf, axis=0)
            info["historical_obs_cur"] = np.stack(self.obs_buf, axis=0)
            self.act_buf.append(action)
            self.obs_buf.append(obs_next_delayed)
            self.act_buf.pop(0)
            self.obs_buf.pop(0)
            info["historical_act_next"] = np.stack(self.act_buf, axis=0)
            info["historical_obs_next"] = np.stack(self.obs_buf, axis=0)
        elif self.history_num == 0:
            info["historical_act_cur"] = False
            info["historical_obs_cur"] = False
            info["historical_act_next"] = False
            info["historical_obs_next"] = False
        
        return (deepcopy(obs_next_delayed), deepcopy(reward), deepcopy(done), deepcopy(truncated), deepcopy(info))

    def step(self, action):
        """
        make a queue of delayed observations, the size of the queue is delay_steps
        for example, if delay_steps = 2, then the queue is [s_{t-2}, s_{t-1}, s_t]
        for each step, the queue will be updated as [s_{t-1}, s_t, s_{t+1}]
        """
        res = self.env.step(action)
        return self.preprocess_fn(res, action)


# others

class StickyActionWrapper(gym.Wrapper, RecordConstructorArgs):
    """
    Source: https://github.com/openai/random-network-distillation/blob/master/atari_wrappers.py
    """
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p

    def reset(self):
        self.last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs_next_nodelay, reward, done, truncated, info = self.env.step(action)
        return obs_next_nodelay, reward, done, truncated, info

class GaussianNoiseActionWrapper(gym.Wrapper, RecordConstructorArgs):
    def __init__(self, env, noise_fraction):
        super().__init__(env)
        self.noise_fraction = noise_fraction

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        # Calculate the noise scale based on action space range
        action_range = self.action_space.high - self.action_space.low
        noise_scale = self.noise_fraction * action_range * 0.5

        # Add Gaussian noise to the action
        noisy_action = action + np.random.normal(0, noise_scale, size=action.shape)

        # Clip the noisy action to the action range
        clipped_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)

        # Take a step in the environment with the clipped action
        return self.env.step(clipped_action)

class GaussianNoiseObservationWrapper(gym.Wrapper, RecordConstructorArgs):
    def __init__(self, env, gaussian_obs):
        super().__init__(env)
        self.gaussian_obs = gaussian_obs

    def reset(self):
        res = self.env.reset()
        if isinstance(res, tuple): obs, info = res
        else: obs, info = res, {}
        clipped_observation = self.noisify_obs(obs)
        if isinstance(res, tuple):
            return clipped_observation, info
        else:
            return clipped_observation

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4: 
            obs, reward, done, info = res
            truncated = False
        elif len(res) == 5:
            obs, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        clipped_observation = self.noisify_obs(obs)
        # return
        if len(res) == 4: 
            return clipped_observation, reward, done, info
        elif len(res) == 5:
            return clipped_observation, reward, done, truncated, info
    
    def noisify_obs(self, obs):
        # Calculate the noise scale based on action space range
        obs_range = self.observation_space.high - self.observation_space.low
        obs_range = np.where(np.isinf(obs_range), 2., obs_range) # if is inf, set to 2.
        noise_scale = self.gaussian_obs * obs_range * 0.5

        # Add Gaussian noise to the action
        noisy_obs = obs + np.random.normal(0, noise_scale, size=obs.shape)

        # Clip the noisy action to the action range # ! note that for mujoco, there is no limit clip since space is [-inf, inf]
        clipped_observation = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
        return clipped_observation

class NormedObsActWrapper(gym.Wrapper, RecordConstructorArgs):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space_ori = self.observation_space
        self.action_space_ori = self.action_space # ! ?
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=self.observation_space.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=np.float32)
    
    def reset(self):
        res = self.env.reset()
        if isinstance(res, tuple): obs, info = res
        else: obs, info = res, {}
        obs = self.norm_obs(obs)
        if isinstance(res, tuple):
            return obs, info
        else:
            return obs
    
    def step(self, action):
        res = self.env.step(self.denorm_act(action))
        if len(res) == 4: 
            obs, reward, done, info = res
            truncated = False
        elif len(res) == 5:
            obs, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        obs = self.norm_obs(obs)
        # return
        if len(res) == 4: 
            return obs, reward, done, info
        elif len(res) == 5:
            return obs, reward, done, truncated, info
    
    def norm_obs(self, obs):
        obs = (obs - self.observation_space_ori.low) / (self.observation_space_ori.high - self.observation_space_ori.low)
        obs = obs * 2 - 1
        return obs

    def denorm_act(self, act):
        act = (act + 1) / 2
        act = act * (self.action_space_ori.high - self.action_space_ori.low) + self.action_space_ori.low
        return act

class MaxStepWrapper(gym.Wrapper, RecordConstructorArgs):
    def __init__(self, env, max_step):
        super().__init__(env)
        self.max_step_x = max_step
        self.step_cnt_x = 0
    
    def reset(self):
        self.step_cnt_x = 0
        return self.env.reset()
    
    def step(self, action):
        self.step_cnt_x += 1
        res = self.env.step(action)
        if self.step_cnt_x >= self.max_step_x:
            if len(res) == 4: 
                obs, reward, done, info = res
                done = True
            elif len(res) == 5:
                obs, reward, done, truncated, info = res
                truncated = True
            else:
                raise ValueError("Invalid return value from env.step()")
            # return
            if len(res) == 4: 
                return obs, reward, done, info
            elif len(res) == 5:
                return obs, reward, done, truncated, info
        else:
            return res

class MergeObsActWrapper(gym.Wrapper, RecordConstructorArgs):
    """ merge all observation into a single observation
    original observation space: {
        'achieved_goal': Box(-10., 10., (3,), float32),
        'desired_goal': Box(-10., 10., (3,), float32),
        'observation': Box(-inf, inf, (25,), float32)
        ...
    }
    new observation space: Box([-10., -10., ..., -inf, -inf], [10., 10., ..., inf, inf], (31,), float32)
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env.observation_space[key].low for key in env.observation_space.spaces.keys()]),
            high=np.concatenate([env.observation_space[key].high for key in env.observation_space.spaces.keys()]),
            dtype=np.float32
        )

    def reset(self):
        res = self.env.reset()
        if isinstance(res, tuple): obs, info = res
        else: obs, info = res, {}
        obs = self.merge_obs(obs)
        if isinstance(res, tuple):
            return obs, info
        else:
            return obs

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:
            obs, reward, done, info = res
            truncated = False
        elif len(res) == 5:
            obs, reward, done, truncated, info = res
        else:
            raise ValueError("Invalid return value from env.step()")
        obs = self.merge_obs(obs)
        # return
        if len(res) == 4:
            return obs, reward, done, info
        elif len(res) == 5:
            return obs, reward, done, truncated, info
    
    def merge_obs(self, obs):
        return np.concatenate([obs[key] for key in obs.keys()])

class ObsHander:
    def __init__(self):
        self.data = []
    
    def register(self, name, dim_num, space):
        """ space can be a float
        """
        if isinstance(space, float):
            space = gym.spaces.Box(low=-space, high=space, shape=(dim_num,), dtype=np.float32)
        self.data.append((name, dim_num, space))
    
    def distill(self, obs, name):
        """
        obs: (all_dim, ) # where all_dim=sum([dim_num for _, dim_num, _ in self.data])
        name: str
        """
        idx = 0
        for name_, dim_num, _ in self.data:
            if name_ == name:
                return obs[idx:idx+dim_num]
            else:
                idx += dim_num
        raise ValueError("Invalid name {}".format(name))

    def get_all_space(self):
        # all_spc = gym.spaces.Box(
        #     low=np.concatenate([spc.low for _, _, spc in self.data]),
        #     high=np.concatenate([spc.high for _, _, spc in self.data]),
        #     dtype=np.float32
        # )
        # ! use inf since there are bug of that the obs is out of range from panda-gym
        all_spc = gym.spaces.Box(
            low=np.full(sum([dim_num for _, dim_num, _ in self.data]), -np.inf, dtype=np.float32),
            high=np.full(sum([dim_num for _, dim_num, _ in self.data]), np.inf, dtype=np.float32),
            dtype=np.float32
        )
        return all_spc

class BetterResetWrapper(gym.Wrapper, RecordConstructorArgs):
    """
    Extends gym.Wrapper to provide flexible resetting of gym environments. 
    It supports keyword arguments (kwargs) for the reset method, defaulting to 
    the standard reset method if kwargs are not supported by the inner environment.
    """
    def reset(self, **kwargs):
        try:
            # Try to reset the environment with kwargs.
            # This will work if the inner environment's reset method supports kwargs.
            return self.env.reset(**kwargs)
        except TypeError:
            # If a TypeError occurs (likely because kwargs are not supported),
            # fall back to the default reset method without kwargs.
            return self.env.reset()

# utils

class ListAsQueue:
    """ A queue implemented by list, which support indexing.
    """
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []
    
    def put(self, item):
        if len(self.queue) >= self.maxsize:
            self.queue.pop(0)
        self.queue.append(item)
    
    def get(self):
        return self.queue.pop(0)

    def empty(self):
        return len(self.queue) == 0
    
    def full(self):
        return len(self.queue) == self.maxsize
    
    def __getitem__(self, idx):
        return self.queue[idx]

    def __len__(self):
        return len(self.queue)
