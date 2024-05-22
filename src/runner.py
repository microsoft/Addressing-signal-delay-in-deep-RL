# Standard library
from collections.abc import Iterable
from copy import deepcopy
from functools import partial
from time import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import warnings

# Third-party libraries
import numpy as np
import torch
import torch.nn.functional as F
import tianshou
from torch import nn
from torch.distributions import Independent, Normal
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import MLP
from tianshou.exploration import BaseNoise
import wandb
from rich.console import Console
from rich.progress import Progress

# Local modules
import pyrootutils
pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import utils
from src.tianshou.policy import DDPGPolicy

# Setup and configurations
warnings.filterwarnings('ignore')

# utils

def forward_with_preinput(
		input,
		preinput,
		net,
		stage,
		forward_strategy="once",
	):
	"""
	Args:
		input: (B, T, *)
		preinput: (B, T, *)
		net: a pytorch module (input, state_in) -> (output, state_out)
			where state_in could be None for zero state
		stage: cur or next, would be used to choose the state
	"""
	B, T = input.shape[:2]

	# make state
	if forward_strategy == "once":
		with torch.no_grad():
			_, info = net(preinput, state=None) # (B, T, D*hidden_dim) # note that there is only one hidden available
			hidden_states = info["hidden_all"]
			hidden_dim = net.net.nn.hidden_size
			hidden_states = hidden_states.reshape(B, T, -1, hidden_dim) # (B, T, D, hidden_dim)
			hidden_states = hidden_states.unsqueeze(-2) # (B, T, D, num_layers, hidden_dim)
			hidden_states = hidden_states.reshape(B, T, -1, hidden_dim) # (B, T, D*num_layers, hidden_dim)

	elif forward_strategy == "multi":
		with torch.no_grad():
			# Compute hidden states for each time step
			hidden_states = []
			last_state = None
			for t in range(T):
				input_t = preinput[:, t].unsqueeze(1)
				_, state = net(input_t, state=last_state)
				assert state is not None, "state must not be None, the net should contains RNN when in burnin function"
				hidden_states.append(state["hidden"])
				last_state = state
			# Stack hidden states along the time dimension
			hidden_states = torch.stack(hidden_states, dim=1) # (B, T, D*num_layers, hidden_dim)

	# select the state
	hidden_states = torch.cat([torch.zeros_like(hidden_states[:,0:1]), hidden_states], dim=1)
	if stage == "cur":
		hidden_states = hidden_states[:, :-1]
	elif stage == "next":
		hidden_states = hidden_states[:, 1:]
	
	# hidden_states: (B, T, D*num_layers, hidden_dim)
	input = input.reshape(B*T, 1, -1)
	hidden_states = hidden_states.reshape(B*T, *hidden_states.shape[2:]) # (B*T, D*num_layers, hidden_dim)
	# hidden_states = hidden_states.permute(1, 0, 2) # (D*num_layers, B*T, hidden_dim) # the inner layer would do the transpose
	output, _ = net(input, {"hidden":hidden_states}) # (B*T, out_dim)
	output = output.reshape(B, T, -1)
	return output

def forward_with_burnin(
	input,    
	burnin_input,
	net,
	remaster_mode,
	remaster_mask,
	):
	"""
	Args:
		input: (B, T, *)
		burnin_input: (B, Tb, *)
		net: a pytorch module (input, state_in) -> (output, state_out)
			where state_in could be None for zero state
		remaster_mask: (B, T)
		ps. assume the last one of burnin is corresponding to the first in the cur of input
	Return:
		state
	"""
	B, T = input.shape[:2]
	B_b, Tb = burnin_input.shape[:2]
	assert B == B_b, "Batch sizes must match for input and burnin_input"

	if remaster_mask is None:
		remaster_mask = torch.ones(B, T, dtype=torch.bool, device=input.device)

	# Perform burn-in to obtain initial state
	state_cur, state_next = burnin_to_get_state(burnin_input, remaster_mask, net)

	# Choose initial state depending on remaster_mode
	initial_state = state_cur if remaster_mode == "cur" else state_next

	# Forward pass with the initial state from burn-in
	output, _ = net(input, initial_state)

	return output

def burnin_to_get_state(input, mask, net, forward_strategy="once"):
	""" 
	process input to the net to get the final state after the final valid mask

	Structure:
		... ,_ , 1, 2, 3, 4, 5, 6, 7, 8, 9, ... Memory Buffer
		... ,_ , _, 3, 4, 5, _, _, _, _, ... main sequence
		... ,_ , 2, 3, _, _, _, _, _, _, ... burnin
		... ,0 , 1, 1, _, _, _, _, _, _, ... burn in mask
		after shifting
		burnin      = [2, 3, 0]
		burnin_mask = [1, 1, 0]
		so the full state would be [2, 3, 0]
		so state_cur should be the state after 2 # after second last 1 in mask
		and state_next should be the state after 2, 3 # = after all 1 in mask
		e.g. in this case, the index should be 1 for state_cur and 2 for state_next
		ps. special case:
			if all invalid: impossible, since the main seq must be valid
				so the last one of burn in must be valid
			if only one (the one must be the first element),    
				then state=1, second_last_burnin_idx=0
				the the one must be
	Args:
		input: (B, T, in_dim_of_net)
		mask: (B, T)
		net:
			forward a_in and return out, state
			where state = {
				"hidden": (B, num_layers, hidden_dim)
			e.g. net(burnin_batch.a_in,state=None)[1]["hidden"].shape
	Returns:
		state_cur: (B, num_layers, hidden_dim)
		state_next: (B, num_layers, hidden_dim)
		for state_cur, it should be the state after the last second burnin (since the last one the start of the main sequence) 
		for state_next, it should be the state after the last burnin
	"""
	assert len(input.shape) == 3 and len(mask.shape) == 2
	B, T = input.shape[:2]
	
	if forward_strategy == "once":
		with torch.no_grad():
			_, info = net(input, state=None) # (B, T, D*hidden_dim) # note that there is only one hidden available
			hidden_states = info["hidden_all"]
			hidden_dim = net.net.nn.hidden_size
			hidden_states = hidden_states.reshape(B, T, -1, hidden_dim) # (B, T, D, hidden_dim)
			hidden_states = hidden_states.unsqueeze(-2) # (B, T, D, num_layers, hidden_dim)
			hidden_states = hidden_states.reshape(B, T, -1, hidden_dim) # (B, T, D*num_layers, hidden_dim)

	elif forward_strategy == "multi":
		with torch.no_grad():
			# Compute hidden states for each time step
			hidden_states = []
			last_state = None
			for t in range(T):
				input_t = input[:, t].unsqueeze(1)
				_, state = net(input_t, state=last_state)
				assert state is not None, "state must not be None, the net should contains RNN when in burnin function"
				hidden_states.append(state["hidden"])
				last_state = state

		# Stack hidden states along the time dimension
		hidden_states = torch.stack(hidden_states, dim=1) # (B, T, D*num_layers, hidden_dim)

	# Identify the last and second last burn-in indices
	burnin_mask = mask == 1
	burnin_cumsum = burnin_mask.cumsum(dim=-1)
	burnin_cnt = burnin_cumsum.max(dim=-1).values
	assert (burnin_cnt > 0).all(), "all must > 0, be valid"
	
	last_burnin_idx = burnin_cnt - 1 # (B,)
	second_last_burnin_idx = burnin_cnt - 2 # (B,)

	# Extract states for the last second and last burn-in steps
	state_cur = {"hidden": hidden_states[torch.arange(B), second_last_burnin_idx, :, :]}
	state_next = {"hidden": hidden_states[torch.arange(B), last_burnin_idx, :, :]}

	# Initialize zero states for cases where burn-in index is less than 0
	zero_state = torch.zeros_like(hidden_states[:, 0])

	# Replace the states with zero state where burn-in index is less than 0
	for k, v in state_cur.items():
		state_cur[k] = torch.where(second_last_burnin_idx.unsqueeze(-1).unsqueeze(-1) < 0, zero_state, v)
	for k, v in state_next.items():
		state_next[k] = torch.where(last_burnin_idx.unsqueeze(-1).unsqueeze(-1) < 0, zero_state, v)

	return state_cur, state_next

def distill_state(state, key_maps):
	"""
	e.g. key_maps = {"hidden_pred_net_encoder": "hidden_encoder", "hidden_pred_net_decoder": "hidden_decoder"}
	"""
	if state is None: return None
	res = {}
	for k, v in key_maps.items():
		if k in state: res[v] = state[k]
	return res if res else None

def update_state(state_dict_res, state_for_reference, key_maps):
	"""
	e.g. key_maps = {"hidden_encoder": "hidden_pred_net_encoder", "hidden_decoder": "hidden_pred_net_decoder"}
	"""
	assert state_dict_res is not None, "state should not be None"
	res = state_dict_res
	if state_for_reference is None: state_for_reference = {}
	for k, v in key_maps.items():
		if k in state_for_reference: res[v] = state_for_reference[k]
	return res

def kl_divergence(mu1, logvar1, mu2, logvar2):
	"""
	mu1, logvar1: mean and log variance of the first Gaussian distribution
	mu2, logvar2: mean and log variance of the second Gaussian distribution
	input:
		mu1, mu2: (B, K)
		logvar1, logvar2: (B, K)
	output:
		kl: (B, )
	"""
	kl = 0.5 * (
		logvar2 - logvar1 + \
		(torch.exp(logvar1) + (mu1 - mu2).pow(2)) \
		/ torch.exp(logvar2) \
		- 1
		)
	return kl.sum(dim=-1)

def apply_mask(tensor, mask):
	# Ensure the mask tensor and the input tensor have the same starting dimensions
	assert mask.shape == tensor.shape[:len(mask.shape)], "Mask and tensor dimensions mismatch"

	# Reshape the mask tensor to have the same number of dimensions as the input tensor
	mask_reshaped = mask.view(*mask.shape, *([1] * (tensor.dim() - len(mask.shape))))

	# Multiply the input tensor by the reshaped mask
	masked_tensor = tensor * mask_reshaped

	return masked_tensor

class DummyNet(nn.Module):
	"""Return input as output."""
	def __init__(self, **kwargs):
		super().__init__()
		# set all kwargs as self.xxx
		for k, v in kwargs.items():
			setattr(self, k, v)
		
	def forward(self, x):
		return x

class ReplayBuffer(tianshou.data.ReplayBuffer):
	def __init__(
		self,
		size: int,
		stack_num: int = 1,
		ignore_obs_next: bool = False,
		save_only_last_obs: bool = False,
		sample_avail: bool = False,
		seq_len: int = 0, # for ReMaster
		**kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
	) -> None:
		super().__init__(size, stack_num, ignore_obs_next, save_only_last_obs, sample_avail, **kwargs)
		assert seq_len > 0, "seq_len should be non-negative"
		self._seq_len = seq_len
		self._remaster_idx_buf = np.array([None for _ in range(self.maxsize*10)])
		self._remaster_idx_ptr = 0 + (self._seq_len - 1) # init gap

	
	def add(
		self,
		batch: Batch,
		buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		idx = self._index
		res = super().add(batch, buffer_ids)
		batch = self[idx]
		self._remaster_idx_buf[self._remaster_idx_ptr] = idx
		self._remaster_idx_ptr += 1
		if batch.done:
			self._remaster_idx_ptr += (self._seq_len - 1)
		return res

	def sample_indices(self, batch_size: int) -> np.ndarray:
		# raise ValueError("For ReMaster Buffer, please use sample_indices_remaster() instead")
		return super().sample_indices(batch_size)

	def sample_indices_remaster(self, batch_size: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Args:
			batch_size: number of sequences to sample
			seq_len: length of each sequence
		"""
		assert seq_len == self._seq_len, "seq_len should be equal to self._seq_len"
		idx_start = np.random.randint(0, self._remaster_idx_ptr - seq_len, batch_size) # B
		idxes_buf, idxes_remaster, valid_mask = self._remaster_expand(idx_start, seq_len, direction="right")
		assert np.all(valid_mask.sum(axis=1) > 0), "There is at least one sequence without valid index"
		return idxes_buf, idxes_remaster, valid_mask

	def create_burnin_pair(self, idxes_remaster, burnin_len):
		"""create burnin pair with main batch idxes
		hint: walking left from idxes
		"""
		assert len(idxes_remaster.shape) == 2, "idxes should be 2-dim, should support more later"
		return self._remaster_expand(idxes_remaster[:,0], burnin_len, direction="left", contain_head=True)
	
	def _remaster_expand(self, idx_start, seq_len, direction="left", contain_head=True):
		""" expand the idxes to a sequence of idxes
		idx_start: (*, )
		return:
			idxes: (*, seq_len)
			remater_idxes: (*, seq_len)
			valid_mask: (*, seq_len)
		"""
		assert self._remaster_idx_ptr < len(self._remaster_idx_buf), "ReMaster Buffer is full, not as expected, should  write related code"
		assert len(idx_start.shape) == 1, "idx_start should be 1-dim, should support more later"
		batch_size = len(idx_start)

		buf = self._remaster_idx_buf
		if direction == "right":
			if contain_head:
				idx_stack_remaster = np.array([np.arange(idx_start[i], idx_start[i] + seq_len) for i in range(len(idx_start))]) # B, L
			else:
				idx_stack_remaster = np.array([np.arange(idx_start[i] + 1, idx_start[i] + seq_len + 1) for i in range(len(idx_start))])
		elif direction == "left":
			if contain_head:
				idx_stack_remaster = np.array([np.arange(idx_start[i] - seq_len + 1, idx_start[i] + 1) for i in range(len(idx_start))])
			else:
				idx_stack_remaster = np.array([np.arange(idx_start[i] - seq_len, idx_start[i]) for i in range(len(idx_start))])
		else:
			raise ValueError("Invalid direction value for ReMaster Buffer {}".format(direction))
		valid_mask = buf[idx_stack_remaster] != None # B, L

		# Find the first valid index in each sequence
		first_valid_indices = np.argmax(valid_mask, axis=1)

		# Shift the sequences and masks to the left to start with the first valid index
		idx_stack_remaster = np.array([np.roll(idx_stack_remaster[i], -first_valid_indices[i]) for i in range(batch_size)])
		valid_mask_shifted = np.array([np.roll(valid_mask[i], -first_valid_indices[i]) for i in range(batch_size)])

		# Turn to the original indices (replace None with 0)
		idx_stack_buf = buf[idx_stack_remaster]
		idx_stack_buf[~valid_mask_shifted] = 0
		return idx_stack_buf.astype(np.int32), idx_stack_remaster.astype(np.int32), valid_mask_shifted


	def get(
		self,
		index: Union[int, List[int], np.ndarray],
		key: str,
		default_value: Any = None,
		stack_num: Optional[int] = None,
	) -> Union[Batch, np.ndarray]:
		return super().get(index, key, default_value, stack_num)

	def reset(self, keep_statistics: bool = False) -> None:
		# raise ValueError("ReplayBuffer.reset() is not supported for ReMaster Buffer")
		return super().reset(keep_statistics)

class EnvCollector:
	"""
	Use policy to collect data from env.
	This collector will continue from the last state of the env.
	"""

	def __init__(self, env):
		self.env = env
		# from minari import DataCollectorV0 as DataCollector
		self.env_loop = self.create_env_loop()

	def collect(self, act_func, n_step=None, n_episode=None, env_max_step=5000, reset=False, progress_bar=None, rich_progress=None):
		"""
		Return 
			res: a list of Batch(obs, act, rew, done, obs_next, info).
			# info: {
			# 	"rews": list of total rewards of each episode,
			# }
		Policy can be function or string "random". 
			function: 
				input: batch, state output: a, state
				state is {"hidden": xxx, "hidden_pre": xxx}
		n_step and n_episode should be provided one and only one.
		Will continue from the last state of the env if reset=False.
		"""
		assert isinstance(act_func, (str, Callable)), "act_func should be a function or string 'random'"
		assert (n_step is None) ^ (n_episode is None), "n_step and n_episode should be provided one and only one"
		if progress_bar is not None: assert rich_progress is not None, "rich process must be provided to display process bar"

		if reset == True: self.to_reset = True
		self.act_func = act_func
		self.env_max_step = env_max_step
		res_list = []
		res_info = {
			"rew_sum_list": [],
			"ep_len_list": []
		}

		step_cnt = 0
		episode_cnt = 0
		finish_flag = False
		if progress_bar is not None:
			progress = rich_progress
			task = progress.add_task(progress_bar, total=n_episode if n_episode is not None else n_step)
		while not finish_flag:
			batch, env_loop_info = next(self.env_loop)
			res_list.append(batch)

			if (batch.terminated or batch.truncated).any(): 
				episode_cnt += 1
				if progress_bar is not None: progress.update(task, advance=1)
				res_info["rew_sum_list"].append(env_loop_info["rew_sum"])
				res_info["ep_len_list"].append(env_loop_info["ep_len"])

			if n_step is not None:
				step_cnt += 1
				if progress_bar is not None: progress.update(task, advance=1)
				finish_flag = step_cnt >= n_step
			elif n_episode is not None:
				finish_flag = episode_cnt >= n_episode
			
		if progress_bar is not None: progress.remove_task(task)
		return res_list, res_info

	def create_env_loop(self):
		"""
		Infinite loop, yield a Batch(obs, act, rew, done, obs_next, info).
		Will restart from 0 and return Batch(s_0, ...) if self.to_reset = True.
		"""
		while True:
			env_step_cur = 0
			rew_sum_cur = 0.
			s, info = self.env.reset()
			info["is_first_step"] = True
			last_state = None

			while True:
				a, last_state = self._select_action(Batch(obs=s, info=info), last_state)
				# if a is tensor, turn to numpy array
				if isinstance(a, torch.Tensor):
					a = a.detach()
					if a.device != torch.device("cpu"):
						a = a.cpu()
					a = a.numpy()
				s_, r, terminated, truncated, info = self.env.step(a)
				rew_sum_cur += r

				truncated = truncated or (env_step_cur == self.env_max_step)
				batch = Batch(obs=s, act=a, rew=r, terminated=terminated, truncated=truncated, obs_next=s_, info=info)
				
				yield batch, {
					"rew_sum": rew_sum_cur,
					"ep_len": env_step_cur,
				}

				if self.to_reset:
					self.to_reset = False
					break

				if terminated or truncated:
					break

				env_step_cur += 1
				s = s_

	def _select_action(self, s, state):
		if self.act_func == "random":
			return self.env.action_space.sample(), None
		else:
			return self.act_func(s, state)

	def reset(self):
		self.to_reset = True

class WaybabaRecorder:
	"""
	store all digit values during training and render in different ways.
	self.data = {
		"name": {
			"value": [],
			"show_in_progress_bar": True,
			"upload_to_wandb": False,
			"wandb_logged": False,
		},
		...
	}
	"""
	def __init__(self):
		self.data = {}
	
	def __call__(self, k, v, wandb_=None, progress_bar=None):
		"""
		would update upload_to_wandb and show_in_progress_bar if provided.
		"""
		if k not in self.data: self.data[k] = self._create_new()
		self.data[k]["values"].append(v)
		if progress_bar in [True, False]: self.data[k]["show_in_progress_bar"] = progress_bar
		if wandb_ in [True, False]:  self.data[k]["upload_to_wandb"] = wandb_
		self.data[k]["wandb_logged"] = False

	def upload_to_wandb(self, *args, **kwargs):
		to_upload = {}
		for k, v in self.data.items():
			if v["upload_to_wandb"] and not v["wandb_logged"] and len(v["values"]) > 0:
				to_upload[k] = v["values"][-1]
				self.data[k]["wandb_logged"] = True
		if len(to_upload) > 0:
			wandb.log(to_upload, *args, **kwargs)

	def to_progress_bar_description(self):
		return self.__str__()

	def _create_new(self):
		return {
			"values": [],
			"show_in_progress_bar": True,
			"upload_to_wandb": True,
			"wandb_logged": False,
		}
	
	def __str__(self):
		info_dict = {
			k: v["values"][-1] for k, v in \
			sorted(self.data.items(), key=lambda item: item[0]) \
			if v["show_in_progress_bar"] and len(v["values"]) > 0
		}
		for k, v in info_dict.items():
			if type(v) == int:
				info_dict[k] = str(v)
			elif type(v) == float:
				info_dict[k] = '{:.2f}'.format(v)
			else:
				info_dict[k] = '{:.2f}'.format(v)

		# Find the maximum length of keys and values
		max_key_length = max(len(k) for k in info_dict.keys())
		max_value_length = max(len(v) for v in info_dict.values())

		# Align keys to the left and values to the right
		aligned_info = []
		for k, v in info_dict.items():
			left_aligned_key = k.ljust(max_key_length)
			right_aligned_value = v.rjust(max_value_length)
			aligned_info.append(f"{left_aligned_key} {right_aligned_value}")

		return "\n".join(aligned_info)


"""Tinashou"""

class AsyncACDDPGPolicy(DDPGPolicy):
	@staticmethod
	def _mse_optimizer(
		batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		# current_q = critic(batch.obs, batch.act).flatten()
		current_q = critic(batch.info["obs_nodelay"], batch.act).flatten()
		target_q = batch.returns.flatten()
		td = current_q - target_q
		# critic_loss = F.mse_loss(current_q1, target_q)
		critic_loss = (td.pow(2) * weight).mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss

	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# critic
		td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
		batch.weight = td  # prio-buffer
		# actor
		actor_loss = -self.critic(batch.info["obs_nodelay"], self(batch).act).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
		self.sync_weight()
		return {
			"loss/actor": actor_loss.item(),
			"loss/critic": critic_loss.item(),
		}

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		next_batch = buffer[indices + 1]  # next_batch.obs: s_{t+n+1}
		next_batch.c_in_curur = next_batch.info["obs_cur"]
		# obs_next_result = self(batch, input="obs_cur")
		obs_next_result = self(next_batch, input="obs_cur")
		act_ = obs_next_result.act
		target_q = torch.min(
			# self.critic1_old(batch.obs_next, act_),
			# self.critic2_old(batch.obs_next, act_),
			self.critic1_old(next_batch.c_in_curur, act_),
			self.critic2_old(next_batch.c_in_curur, act_),
		) - self._alpha * obs_next_result.log_prob
		return target_q

class SACPolicy(DDPGPolicy):
	"""Implementation of Soft Actor-Critic. arXiv:1812.05905.

	:param torch.nn.Module actor: the actor network following the rules in
		:class:`~tianshou.policy.BasePolicy`. (s -> logits)
	:param torch.optim.Optimizer actor_optim: the optimizer for actor network.
	:param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic1_optim: the optimizer for the first
		critic network.
	:param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic2_optim: the optimizer for the second
		critic network.
	:param float tau: param for soft update of the target network. Default to 0.005.
	:param float gamma: discount factor, in [0, 1]. Default to 0.99.
	:param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
		regularization coefficient. Default to 0.2.
		If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
		alpha is automatically tuned.
	:param bool reward_normalization: normalize the reward to Normal(0, 1).
		Default to False.
	:param BaseNoise exploration_noise: add a noise to action for exploration.
		Default to None. This is useful when solving hard-exploration problem.
	:param bool deterministic_eval: whether to use deterministic action (mean
		of Gaussian policy) instead of stochastic action sampled by the policy.
		Default to True.
	:param bool action_scaling: whether to map actions from range [-1, 1] to range
		[action_spaces.low, action_spaces.high]. Default to True.
	:param str action_bound_method: method to bound action to range [-1, 1], can be
		either "clip" (for simply clipping the action) or empty string for no bounding.
		Default to "clip".
	:param Optional[gym.Space] action_space: env's action space, mandatory if you want
		to use option "action_scaling" or "action_bound_method". Default to None.
	:param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
		optimizer in each policy.update(). Default to None (no lr_scheduler).

	.. seealso::

		Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
		explanation.
	"""

	def __init__(
		self,
		actor: torch.nn.Module,
		actor_optim: torch.optim.Optimizer,
		critic1: torch.nn.Module,
		critic1_optim: torch.optim.Optimizer,
		critic2: torch.nn.Module,
		critic2_optim: torch.optim.Optimizer,
		tau: float = 0.005, # TODO use hyperparameter in the paper
		gamma: float = 0.99,
		alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
		reward_normalization: bool = False,
		estimation_step: int = 1,
		exploration_noise: Optional[BaseNoise] = None,
		deterministic_eval: bool = True,
		**kwargs: Any,
	) -> None:
		super().__init__(
			None, None, None, None, tau, gamma, exploration_noise,
			reward_normalization, estimation_step, **kwargs
		)
		self.actor, self.actor_optim = actor, actor_optim
		self.critic1, self.critic1_old = critic1, deepcopy(critic1)
		self.critic1_old.eval()
		self.critic1_optim = critic1_optim
		self.critic2, self.critic2_old = critic2, deepcopy(critic2)
		self.critic2_old.eval()
		self.critic2_optim = critic2_optim

		if isinstance(alpha, tuple):
			self._is_auto_alpha = True
			self._target_entropy, self._log_alpha, self._alpha_optim = alpha
			if type(self._target_entropy) == str and self._target_entropy == "neg_act_num":
				assert hasattr(self.actor, "act_num"), "actor must have act_num attribute"
				act_num = self.actor.act_num
				self._target_entropy = - act_num
			elif type(self._target_entropy) == float:
				pass
			else: 
				raise ValueError("Invalid target entropy type.")
			assert alpha[1].shape == torch.Size([1]) and alpha[1].requires_grad
			self._alpha_optim = self._alpha_optim([alpha[1]])
			self._alpha = self._log_alpha.detach().exp()
		elif isinstance(alpha, float):
			self._is_auto_alpha = False
			self._alpha = alpha
		else: 
			raise ValueError("Invalid alpha type.")

		self._deterministic_eval = deterministic_eval
		self.__eps = np.finfo(np.float32).eps.item()

	def train(self, mode: bool = True) -> "SACPolicy":
		self.training = mode
		self.actor.train(mode)
		self.critic1.train(mode)
		self.critic2.train(mode)
		return self

	def sync_weight(self) -> None:
		self.soft_update(self.critic1_old, self.critic1, self.tau)
		self.soft_update(self.critic2_old, self.critic2, self.tau)

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		obs_next_result = self(batch, input="obs_next")
		act_ = obs_next_result.act
		target_q = torch.min(
			self.critic1_old(batch.obs_next, act_),
			self.critic2_old(batch.obs_next, act_),
		) - self._alpha * obs_next_result.log_prob
		return target_q

	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# critic 1&2
		td1, critic1_loss = self._mse_optimizer(
			batch, self.critic1, self.critic1_optim
		)
		td2, critic2_loss = self._mse_optimizer(
			batch, self.critic2, self.critic2_optim
		)
		batch.weight = (td1 + td2) / 2.0  # prio-buffer

		# actor
		obs_result = self(batch)
		act = obs_result.act
		current_q1a = self.critic1(batch.obs, act).flatten()
		current_q2a = self.critic2(batch.obs, act).flatten()
		actor_loss = (
			self._alpha * obs_result.log_prob.flatten() -
			torch.min(current_q1a, current_q2a)
		).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()

		if self._is_auto_alpha: # TODO auto alpha
			log_prob = obs_result.log_prob.detach() + self._target_entropy
			# please take a look at issue #258 if you'd like to change this line
			alpha_loss = -(self._log_alpha * log_prob).mean()
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self._alpha = self._log_alpha.detach().exp()

		self.sync_weight()

		result = {
			"loss/actor": actor_loss.item(),
			"loss/critic1": critic1_loss.item(),
			"loss/critic2": critic2_loss.item(),
		}
		if self._is_auto_alpha:
			result["loss/alpha"] = alpha_loss.item()
			result["alpha"] = self._alpha.item()  # type: ignore

		return result

class TD3Policy(DDPGPolicy):
	"""Implementation of TD3, arXiv:1802.09477.

	:param torch.nn.Module actor: the actor network following the rules in
		:class:`~tianshou.policy.BasePolicy`. (s -> logits)
	:param torch.optim.Optimizer actor_optim: the optimizer for actor network.
	:param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic1_optim: the optimizer for the first
		critic network.
	:param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic2_optim: the optimizer for the second
		critic network.
	:param float tau: param for soft update of the target network. Default to 0.005.
	:param float gamma: discount factor, in [0, 1]. Default to 0.99.
	:param float exploration_noise: the exploration noise, add to the action.
		Default to ``GaussianNoise(sigma=0.1)``
	:param float policy_noise: the noise used in updating policy network.
		Default to 0.2.
	:param int update_actor_freq: the update frequency of actor network.
		Default to 2.
	:param float noise_clip: the clipping range used in updating policy network.
		Default to 0.5.
	:param bool reward_normalization: normalize the reward to Normal(0, 1).
		Default to False.
	:param bool action_scaling: whether to map actions from range [-1, 1] to range
		[action_spaces.low, action_spaces.high]. Default to True.
	:param str action_bound_method: method to bound action to range [-1, 1], can be
		either "clip" (for simply clipping the action) or empty string for no bounding.
		Default to "clip".
	:param Optional[gym.Space] action_space: env's action space, mandatory if you want
		to use option "action_scaling" or "action_bound_method". Default to None.
	:param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
		optimizer in each policy.update(). Default to None (no lr_scheduler).

	.. seealso::

		Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
		explanation.
	"""

	def __init__(
		self,
		actor: torch.nn.Module,
		actor_optim: torch.optim.Optimizer,
		critic1: torch.nn.Module,
		critic1_optim: torch.optim.Optimizer,
		critic2: torch.nn.Module,
		critic2_optim: torch.optim.Optimizer,
		tau: float = 0.005,
		gamma: float = 0.99,
		exploration_noise=None,
		policy_noise: float = 0.2,
		update_actor_freq: int = 2,
		noise_clip: float = 0.5,
		reward_normalization: bool = False,
		estimation_step: int = 1,
		**kwargs: Any,
	) -> None:
		super().__init__(
			actor, actor_optim, None, None, tau, gamma, exploration_noise,
			reward_normalization, estimation_step, **kwargs
		)
		self.critic1, self.critic1_old = critic1, deepcopy(critic1)
		self.critic1_old.eval()
		self.critic1_optim = critic1_optim
		self.critic2, self.critic2_old = critic2, deepcopy(critic2)
		self.critic2_old.eval()
		self.critic2_optim = critic2_optim
		self._policy_noise = policy_noise
		self._freq = update_actor_freq
		self._noise_clip = noise_clip
		self._cnt = 0
		self._last = 0

	def train(self, mode: bool = True) -> "TD3Policy":
		self.training = mode
		self.actor.train(mode)
		self.critic1.train(mode)
		self.critic2.train(mode)
		return self

	def sync_weight(self) -> None:
		self.soft_update(self.critic1_old, self.critic1, self.tau)
		self.soft_update(self.critic2_old, self.critic2, self.tau)
		self.soft_update(self.actor_old, self.actor, self.tau)

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs: s_{t+n}
		act_ = self(batch, model="actor_old", input="obs_next").act
		noise = torch.randn(size=act_.shape, device=act_.device) * self._policy_noise
		if self._noise_clip > 0.0:
			noise = noise.clamp(-self._noise_clip, self._noise_clip)
		act_ += noise
		target_q = torch.min(
			self.critic1_old(batch.obs_next, act_),
			self.critic2_old(batch.obs_next, act_),
		)
		return target_q

	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# critic 1&2
		td1, critic1_loss = self._mse_optimizer(
			batch, self.critic1, self.critic1_optim
		)
		td2, critic2_loss = self._mse_optimizer(
			batch, self.critic2, self.critic2_optim
		)
		batch.weight = (td1 + td2) / 2.0  # prio-buffer

		# actor
		if self._cnt % self._freq == 0:
			actor_loss = -self.critic1(batch.obs, self(batch, eps=0.0).act).mean()
			self.actor_optim.zero_grad()
			actor_loss.backward()
			self._last = actor_loss.item()
			self.actor_optim.step()
			self.sync_weight()
		self._cnt += 1
		return {
			"loss/actor": self._last,
			"loss/critic1": critic1_loss.item(),
			"loss/critic2": critic2_loss.item(),
		}

	def forward(
		self,
		batch: Batch,
		state: Optional[Union[dict, Batch, np.ndarray]] = None,
		model: str = "actor",
		input: str = "obs",
		**kwargs: Any,
	) -> Batch:
		"""Compute action over the given batch data.

		:return: A :class:`~tianshou.data.Batch` which has 2 keys:

			* ``act`` the action.
			* ``state`` the hidden state.

		.. seealso::

			Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
			more detailed explanation.
		"""
		model = getattr(self, model)
		obs = batch[input]
		actions, hidden = model(obs, state=state, info=batch.info)
		return Batch(act=actions, state=hidden)
	
class TianshouTD3Wrapper(TD3Policy):
	def __init__(self, *args, **kwargs):
		self.global_cfg = kwargs.pop("global_cfg")
		self.state_space = kwargs.pop("state_space")
		super().__init__(*args, **kwargs)

class Critic(nn.Module):
	"""Simple critic network. Will create an actor operated in continuous \
	action space with structure of preprocess_net ---> 1(q value).

	:param preprocess_net: a self-defined preprocess_net which output a
		flattened hidden state.
	:param hidden_sizes: a sequence of int for constructing the MLP after
		preprocess_net. Default to empty sequence (where the MLP now contains
		only a single linear layer).
	:param int preprocess_net_output_dim: the output dimension of
		preprocess_net.
	:param linear_layer: use this module as linear layer. Default to nn.Linear.
	:param bool flatten_input: whether to flatten input data for the last layer.
		Default to True.

	For advanced usage (how to customize the network), please refer to
	:ref:`build_the_network`.

	.. seealso::

		Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
		of how preprocess_net is suggested to be defined.
	"""

	def __init__(
		self,
		preprocess_net: nn.Module,
		hidden_sizes: Sequence[int] = (),
		device: Union[str, int, torch.device] = "cpu",
		preprocess_net_output_dim: Optional[int] = None,
		linear_layer: Type[nn.Linear] = nn.Linear,
		flatten_input: bool = True,
	) -> None:
		super().__init__()
		self.device = device
		self.preprocess = preprocess_net
		self.output_dim = 1
		input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
		self.last = MLP(
			input_dim,  # type: ignore
			1,
			hidden_sizes,
			device=self.device,
			linear_layer=linear_layer,
			flatten_input=flatten_input,
		)

	def forward(
		self,
		obs: Union[np.ndarray, torch.Tensor],
		state: Optional[Dict[str, torch.Tensor]] = None,
	) -> torch.Tensor:
		"""Mapping: (s, a) -> logits -> Q(s, a)."""
		obs = torch.as_tensor(
			obs,
			device=self.device,
			dtype=torch.float32,
		).flatten(1)
		if act is not None:
			act = torch.as_tensor(
				act,
				device=self.device,
				dtype=torch.float32,
			).flatten(1)
			obs = torch.cat([obs, act], dim=1)
		logits, hidden = self.preprocess(obs)
		logits = self.last(logits)
		return logits

# net
class RNN_MLP_Net(nn.Module):
	""" RNNS with MLPs as the core network
	ps. assume input is one dim
	ps. head_num = 1 for critic
	"""
	def __init__(
		self,
		input_dim: int,
		output_dim: int,
		head_num: int,
		device: str,
		rnn_layer_num: int, # in config
		rnn_hidden_layer_size: int, # in config
		mlp_hidden_sizes: Sequence[int], # in config
		activation: str, # in config
		mlp_softmax: bool,  # TODO add # in config
		dropout: float = None, # in config
		bidirectional: bool = False, # in config
	):
		super().__init__()
		self.bidirectional = bidirectional
		self.device = device
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.rnn_layer_num = rnn_layer_num
		self.rnn_hidden_layer_size = rnn_hidden_layer_size
		self.mlp_hidden_sizes = mlp_hidden_sizes
		self.dropout = dropout

		if activation == "relu":
			self.activation = nn.ReLU
		elif activation == "tanh":
			self.activation = nn.Tanh
		elif activation == "sigmoid":
			self.activation = nn.Sigmoid
		else:
			raise NotImplementedError

		# build rnn
		if rnn_layer_num:
			self.nn = nn.GRU(
				input_size=input_dim,
				hidden_size=rnn_hidden_layer_size,
				num_layers=rnn_layer_num,
				batch_first=True,
				bidirectional=bidirectional,
			)
		else:
			self.nn = DummyNet(input_dim=input_dim, input_size=input_dim)
		
		# build mlp
		assert len(mlp_hidden_sizes) > 0, "mlp_hidden_sizes must be > 0"
		before_head_mlp_hidden_sizes = mlp_hidden_sizes[:-1]
		if rnn_layer_num:
			if bidirectional:
				mlp_input_dim = rnn_hidden_layer_size * 2
			else:
				mlp_input_dim = rnn_hidden_layer_size
		else:
			mlp_input_dim = input_dim
		self.mlp_before_head = []
		self.mlp_before_head.append(MLP(
			mlp_input_dim,
			mlp_hidden_sizes[-1],
			before_head_mlp_hidden_sizes,
			device=self.device,
			activation=self.activation,
		))
		self.mlp_before_head.append(self.activation())
		if self.dropout:
			self.mlp_before_head.append(nn.Dropout(self.dropout))
		self.heads = []
		for _ in range(head_num):
			head = MLP(
				mlp_hidden_sizes[-1],
				output_dim,
				hidden_sizes=(),
				device=self.device,
			)
			self.heads.append(head.to(self.device))
		
		self.mlp_before_head = nn.Sequential(*self.mlp_before_head)
		self.heads = nn.ModuleList(self.heads)
	
	def forward(
		self,
		obs: Union[np.ndarray, torch.Tensor],
		state: Optional[Dict[str, torch.Tensor]] = None,
		info: Dict[str, Any] = {},
		):
		"""
		input
		"""
		obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
		### forward rnn
		# obs [bsz, len, dim] (training) or [bsz, dim] (evaluation)
		# In short, the tensor's shape in training phase is longer than which
		# in evaluation phase. 
		# TODO check the format when inputed, to make this neat
		if self.rnn_layer_num: 
			to_unsqueeze_from_1 = False
			to_unsqueeze = False
			if len(obs.shape) == 1: 
				obs = obs.unsqueeze(0)
				to_unsqueeze_from_1 = True
			assert len(obs.shape) == 3 or len(obs.shape) == 2, "obs.shape: {}".format(obs.shape)
			
			if len(obs.shape) == 2: 
				to_unsqueeze = True
				obs = obs.unsqueeze(-2) # make seq_len dim
			B, L, D = obs.shape
			self.nn.flatten_parameters()
			if state is None or state["hidden"] is None:
				# first step of online or offline
				hidden = torch.zeros(self.rnn_layer_num*(2 if self.bidirectional else 1), B, self.rnn_hidden_layer_size, device=self.device)
				after_rnn, hidden = self.nn(obs, hidden)
			else: 
				# normal step of online
				after_rnn, hidden = self.nn(obs, state["hidden"].transpose(0, 1).contiguous())
			if to_unsqueeze: after_rnn = after_rnn.squeeze(-2)
			if to_unsqueeze_from_1: after_rnn = after_rnn.squeeze(0)
		else: # skip rnn
			after_rnn = obs
		
		### forward mlp
		before_head = self.flatten_foward(self.mlp_before_head, after_rnn)
		
		### forward head
		outputs = [self.flatten_foward(head, before_head) for head in self.heads]

		return outputs, {
			"hidden": hidden.transpose(0, 1).detach(),
			"hidden_all": after_rnn.detach(),
		} if self.rnn_layer_num else None

	def flatten_foward(self, net, input):
		"""Flatten input for mlp forward, then reshape output to original shape.
		input: 
			mlp: a mlp module
			after_rnn: tensor [*, D_in]
		output:
			tensor [*, D_out]
		"""
		# flatten
		pre_sz, dim_in = input.shape[:-1], input.shape[-1]
		input = input.reshape(-1, dim_in)
		# forward
		output = net(input)
		# reshape
		dim_out = output.shape[-1]
		output = output.reshape(*pre_sz, dim_out)
		return output

class TransformerNet(nn.Module):
	def __init__(self, input_dim, output_dim, device, head_num, nhead_transformer, num_layers, hidden_layer_size, activation):
		super(TransformerNet, self).__init__()
		self.pre_encoder = nn.Linear(input_dim, hidden_layer_size).to(device)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_layer_size, nhead=nhead_transformer, dim_feedforward=hidden_layer_size, activation=activation)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(device)
		self.decoder = nn.Linear(hidden_layer_size, output_dim).to(device)
		self.heads = []
		for _ in range(head_num):
			head = MLP(
				hidden_layer_size,
				output_dim,
				hidden_sizes=(),
				device=device,
			)
			self.heads.append(head.to(device))
		self.heads = nn.ModuleList(self.heads)
		
	def forward(self, 
		obs: Union[np.ndarray, torch.Tensor],
		state: Optional[Dict[str, torch.Tensor]] = None,
		info: Dict[str, Any] = {},
		):
		x = self.pre_encoder(obs)
		x = self.flatten_2d_forward(self.transformer_encoder, x)
		return [
			self.flatten_foward(head, x[...,0,:]) for head in self.heads # TODO transformer only use one of the feature
		], None

	def flatten_foward(self, net, input):
		"""Flatten input for mlp forward, then reshape output to original shape.
		input: 
			mlp: a mlp module
			after_rnn: tensor [*, D_in]
		output:
			tensor [*, D_out]
		"""
		# flatten
		pre_sz, dim_in = input.shape[:-1], input.shape[-1]
		input = input.reshape(-1, dim_in)
		# forward
		output = net(input)
		# reshape
		dim_out = output.shape[-1]
		output = output.reshape(*pre_sz, dim_out)
		return output
	
	def flatten_2d_forward(self, net, input):
		"""Flatten input for mlp forward, then reshape output to original shape.
		input: 
			mlp: a mlp module
			after_rnn: tensor [*, D_in]
		output:
			tensor [*, D_out]
		"""
		# flatten
		pre_sz, dim_in = input.shape[:-2], input.shape[-2:]
		input = input.reshape(-1, *dim_in)
		# forward
		output = net(input)
		# reshape
		dim_out = output.shape[-2:]
		output = output.reshape(*pre_sz, *dim_out)
		return output
	
class CustomRecurrentCritic(nn.Module):
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		**kwargs,
	) -> None:
		super().__init__()
		self.hps = kwargs
		assert len(state_shape) == 1 and len(action_shape) == 1, "now, only support 1d state and action"
		
		selected_net = self.hps["net_mlp"]
		if self.hps["global_cfg"].critic_input.history_merge_method == "cat_mlp":
			self.input_dim = state_shape[0] + action_shape[0] + action_shape[0] * self.hps["global_cfg"].history_num
			self.output_dim = 1
		elif self.hps["global_cfg"].critic_input.history_merge_method == "stack_rnn":
			if self.hps["global_cfg"].history_num > 0:
				self.input_dim = state_shape[0] + action_shape[0] + action_shape[0]
			else:
				self.input_dim = state_shape[0] + action_shape[0]
			self.output_dim = 1
			selected_net = self.hps["net_rnn"]
		elif self.hps["global_cfg"].critic_input.history_merge_method == "none":
			self.input_dim = state_shape[0] + action_shape[0]
			self.output_dim = 1
		else:
			raise NotImplementedError
		
		bidirectional = True if "bidirectional" in kwargs and kwargs["bidirectional"] else False
		self.net = selected_net(self.input_dim, self.output_dim, device=self.hps["device"], head_num=1, bidirectional=bidirectional)

	def forward(
		self,
		critic_input: Union[np.ndarray, torch.Tensor],
		state: Dict[str, torch.Tensor],
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		"""
		ps. different from actor, the state_ of critic is usually not used, so the pipeline is simpler
		"""
		assert type(info) == dict, "info should be a dict, check whether missing 'info' as act"
		assert type(state) == dict or state is None, "state should be a dict with 'hidden' or None"
		output, state_ = self.net(critic_input, state)
		value = output[0]
		return value, state_

class CustomRecurrentActorProb(nn.Module):
	"""Recurrent version of ActorProb.

	edit log:
		1. add rnn_hidden_layer_size and mlp_hidden_sizes for consecutive processing
			original ActorProb only has one hidden layer after lstm. In the new version, 
			we can customize both the size of RNN hidden layer (with rnn_hidden_layer_size)
			and the size of mlp hidden layer (with mlp_hidden_sizes)
			RNN: rnn_hidden_layer_size * rnn_layer_num
			MLP: mlp_hidden_sizes[0] * mlp_hidden_sizes[1] * ...

	"""
	SIGMA_MIN = -20
	SIGMA_MAX = 2
	
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		**kwargs,
	) -> None:
		super().__init__()
		self.hps = kwargs
		assert len(state_shape) == 1 and len(action_shape) == 1
		self.device = self.hps["device"]
		self.state_shape, self.action_shape = state_shape, action_shape
		self.act_num = action_shape[0]

		selected_net = self.hps["net_mlp"]
		if self.hps["global_cfg"].actor_input.history_merge_method == "cat_mlp":
			if self.hps["global_cfg"].actor_input.obs_pred.turn_on:
				if self.hps["global_cfg"].actor_input.obs_pred.input_type == "feat":
					self.input_dim = self.hps["global_cfg"].actor_input.obs_pred.feat_dim
				elif self.hps["global_cfg"].actor_input.obs_pred.input_type == "obs":
					self.input_dim = state_shape[0]
				else:
					raise ValueError("invalid input_type")
			elif self.hps["global_cfg"].actor_input.obs_encode.turn_on:
				self.input_dim = self.hps["global_cfg"].actor_input.obs_encode.feat_dim
			else:
				self.input_dim = state_shape[0] + action_shape[0] * self.hps["global_cfg"].history_num
			self.output_dim = int(np.prod(action_shape))
		elif self.hps["global_cfg"].actor_input.history_merge_method == "stack_rnn":
			assert self.hps["global_cfg"].history_num == 1, "history_num should be 1 for stack_rnn to cat one more action"
			if self.hps["global_cfg"].actor_input.obs_pred.turn_on:
				if self.hps["global_cfg"].actor_input.obs_pred.input_type == "feat":
					self.input_dim = self.hps["global_cfg"].actor_input.obs_pred.feat_dim
				elif self.hps["global_cfg"].actor_input.obs_pred.input_type == "obs":
					self.input_dim = state_shape[0]
				else:
					raise ValueError("invalid input_type")
			elif self.hps["global_cfg"].actor_input.obs_encode.turn_on:
				self.input_dim = self.hps["global_cfg"].actor_input.obs_encode.feat_dim
			else:
				selected_net = self.hps["net_rnn"]
				if self.hps["global_cfg"].history_num > 0:
					self.input_dim = state_shape[0] + action_shape[0]
				else:
					self.input_dim = state_shape[0]
			self.output_dim = int(np.prod(action_shape))
		elif self.hps["global_cfg"].actor_input.history_merge_method == "transformer":
			if self.hps["global_cfg"].actor_input.obs_pred.turn_on:
				# TODO transformer
				pass
			elif self.hps["global_cfg"].actor_input.obs_encode.turn_on:
				# TODO transformer
				pass
			else:
				selected_net = self.hps["net_transformer"]
				if self.hps["global_cfg"].history_num > 0:
					self.input_dim = state_shape[0] + action_shape[0]
				else:
					self.input_dim = state_shape[0]
			self.output_dim = int(np.prod(action_shape))
		elif self.hps["global_cfg"].actor_input.history_merge_method == "none":
			if self.hps["global_cfg"].actor_input.obs_pred.turn_on:
				if self.hps["global_cfg"].actor_input.obs_pred.input_type == "feat":
					self.input_dim = self.hps["global_cfg"].actor_input.obs_pred.feat_dim
				elif self.hps["global_cfg"].actor_input.obs_pred.input_type == "obs":
					self.input_dim = state_shape[0]
				else:
					raise ValueError("invalid input_type")
			elif self.hps["global_cfg"].actor_input.obs_encode.turn_on:
				self.input_dim = self.hps["global_cfg"].actor_input.obs_encode.feat_dim
			else:
				self.input_dim = state_shape[0]
			self.output_dim = int(np.prod(action_shape))
		else:
			raise NotImplementedError
		

		if self.hps["heads_share_pre_net"]:
			self.net = selected_net(self.input_dim, self.output_dim, device=self.hps["device"], head_num=2)
		else:
			self.mu_net = selected_net(self.input_dim, self.output_dim, device=self.hps["device"], head_num=1)
			self.logsigma_net = selected_net(self.input_dim, self.output_dim, device=self.hps["device"], head_num=1)

	def forward(
		self,
		actor_input: Union[np.ndarray, torch.Tensor],
		state: Dict[str, torch.Tensor],
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		"""Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
		### forward
		if self.hps["heads_share_pre_net"]:
			output, state_ = self.net(actor_input, state)
			mu, logsigma = output
		else:
			if state is not None: raise NotImplementedError("here there are two heads, the state has not be implemented yet")
			(mu,), state_ = self.mu_net(actor_input, state)
			(logsigma,), _ = self.logsigma_net(actor_input, state)
		if not self.hps["unbounded"]:
			mu = self.hps["max_action"] * torch.tanh(mu)
		if self.hps["conditioned_sigma"]:
			sigma = torch.clamp(logsigma, min=self.SIGMA_MIN, max=self.SIGMA_MAX).exp()
		else:
			raise NotImplementedError
		if self.hps["pure_random"]:
			# mu = mu * 1e-10 + torch.normal(mean=0., std=1., size=mu.shape, device=self.device)
			mu = (torch.rand_like(mu, device=self.device) * 2) - 1
			# sigma = torch.ones_like(sigma, device=self.device) * 1e-10
			sigma = sigma * 1e-10
		return (mu, sigma), state_

	def sample_act(self, mu, sigma):
		"""
		Use torch distribution to sample action and return
		act, log_prob
		"""
		ESP = np.finfo(np.float32).eps.item()
		pre_sz = list(mu.size()[:-1])
		normal = Normal(mu, sigma)
		x_t = normal.rsample() # (*, act_dim)
		y_t = torch.tanh(x_t) # (*, act_dim)
		action = y_t * self.hps["max_action"] + 0.0 # TODO use more stable version
		log_prob = normal.log_prob(x_t) # (*, a_dim)
		if self.hps["global_cfg"].debug.dongqi_log_prob_clamp:
			log_prob = log_prob.clamp(-20, 10)
		log_prob = (log_prob - torch.log(1 - y_t.pow(2) + ESP)).sum(dim=-1,keepdim=True) # (*, 1)
		mean = torch.tanh(mu) * self.hps["max_action"]
		return action, log_prob

	def foward_with_burnin(
		self,
		actor_input: Union[np.ndarray, torch.Tensor],
		burnin_input: Union[np.ndarray, torch.Tensor],
		burnin_valid_mask: Union[np.ndarray, torch.Tensor],
		) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		output, burnin_state = self.net(burnin_input, state=None)
		# process state
		# TODO not implemented yet
		in_state = burnin_state
		# forward
		return self.forward(actor_input, in_state)

class ObsPredNet(nn.Module):
	"""
	input delayed state and action, output the non-delayed state
	"""
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		global_cfg: Dict[str, Any],
		**kwargs,
	) -> None:
		super().__init__()
		self.global_cfg = global_cfg
		self.hps = kwargs
		# assert head_num == 1, the rnn layer of decoder is 0
		assert self.hps["net_type"] in ["vae", "mlp", "rnn"], "invalid net_type {}".format(self.hps["net_type"])
		
		# cal input dim
		selected_encoder_net = self.hps["encoder_net_mlp"]
		if self.global_cfg.actor_input.history_merge_method in ["cat_mlp"]:
			self.input_dim = state_shape[0] + action_shape[0] * global_cfg.history_num
		elif self.global_cfg.actor_input.history_merge_method in ["stack_rnn"]:
			selected_encoder_net = self.hps["encoder_net_rnn"]
			if self.hps["global_cfg"].history_num > 0:
				self.input_dim = state_shape[0] + action_shape[0]
			else:
				self.input_dim = state_shape[0]
		elif self.global_cfg.actor_input.history_merge_method in ["none"]:
			self.input_dim = state_shape[0]
		
		# cal output dim
		self.output_dim = state_shape[0]
		self.feat_dim = self.hps["feat_dim"]
		self.encoder_input_dim = self.input_dim
		self.encoder_output_dim = self.feat_dim
		self.decoder_input_dim = self.feat_dim
		self.decoder_output_dim = self.output_dim
		if self.hps["net_type"]=="vae":
			self.encoder_net = selected_encoder_net(self.encoder_input_dim, self.encoder_output_dim, device=self.hps["device"], head_num=2)
		elif self.hps["net_type"]=="mlp":
			self.encoder_net = selected_encoder_net(self.encoder_input_dim, self.encoder_output_dim, device=self.hps["device"], head_num=1)
		elif self.hps["net_type"]=="rnn":
			self.encoder_net = selected_encoder_net(self.encoder_input_dim, self.encoder_output_dim, device=self.hps["device"], head_num=1)
		self.decoder_net = self.hps["decoder_net"](self.decoder_input_dim, self.decoder_output_dim, device=self.hps["device"], head_num=1)
		self.encoder_net.to(self.hps["device"])
		self.decoder_net.to(self.hps["device"])
		
	def forward(
		self,
		input,
		state = None,
		info = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		assert type(info) == dict, "info should be a dict, check whether missing 'info' as act"
		info = {}
		res_state = {}
		
		encoder_outputs, encoder_state = self.encoder_net(input, state=distill_state(state, {"hidden_encoder": "hidden"}))
		if self.hps["net_type"] == "vae":
			mu, logvar = encoder_outputs
			feats = self.vae_sampling(mu, logvar)
			info["mu"] = mu
			info["logvar"] = logvar
		elif self.hps["net_type"] == "mlp":
			feats = encoder_outputs[0]
		elif self.hps["net_type"] == "rnn":
			feats = encoder_outputs[0]
		output, decoder_state = self.decoder_net(feats, state=distill_state(state, {"hidden_decoder": "hidden"}))
		output = output[0]
		res_state = update_state(res_state, encoder_state, {"hidden": "hidden_encoder"})
		res_state = update_state(res_state, decoder_state, {"hidden": "hidden_decoder"})
		info["state"] = res_state if res_state else None
		info["feats"] = feats
		return output, info

	def vae_sampling(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)

class ObsEncodeNet(nn.Module):
	"""
	input delayed state and action, output the non-delayed state
	"""
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		global_cfg: Dict[str, Any],
		**kwargs,
	) -> None:
		super().__init__()
		self.global_cfg = global_cfg
		self.hps = kwargs

		# cal input dim
		selected_encoder_net = self.hps["encoder_net_mlp"]
		if self.global_cfg.actor_input.history_merge_method in ["cat_mlp"]:
			self.normal_encode_dim = state_shape[0] + action_shape[0] * global_cfg.history_num
		elif self.global_cfg.actor_input.history_merge_method in ["stack_rnn"]:
			selected_encoder_net = self.hps["encoder_net_rnn"]
			if self.global_cfg.history_num > 0:
				self.normal_encode_dim = state_shape[0] + action_shape[0]
			else:
				self.normal_encode_dim = state_shape[0]
		elif self.global_cfg.actor_input.history_merge_method in ["none"]:
			self.normal_encode_dim = state_shape[0]
		else:
			raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.actor_input.history_merge_method))
		
		# cal output dim
		self.oracle_encode_dim = state_shape[0]

		self.feat_dim = self.hps["feat_dim"]
		self.normal_encoder_net = selected_encoder_net(self.normal_encode_dim, self.feat_dim, device=self.hps["device"], head_num=2)
		self.oracle_encoder_net = selected_encoder_net(self.oracle_encode_dim, self.feat_dim, device=self.hps["device"], head_num=2)
		self.decoder_net = self.hps["decoder_net"](self.feat_dim, self.oracle_encode_dim, device=self.hps["device"], head_num=1)
		self.normal_encoder_net.to(self.hps["device"])
		self.oracle_encoder_net.to(self.hps["device"])
		self.decoder_net.to(self.hps["device"])
		
	def forward(
		self,
		input: Union[np.ndarray, torch.Tensor],
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		raise ValueError("should call normal_encode or oracle_encode")

	def normal_encode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		return self.net_forward(self.normal_encoder_net, input, info)
	
	def oracle_encode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		return self.net_forward(self.oracle_encoder_net, input, info)
	
	def net_forward(self, net, input, info):
		info = {}
		(mu, logvar), state_ = net(input)
		# feats = self.vae_sampling(mu, logvar)
		feats = self.torch_sampling(mu, logvar)
		info["mu"] = mu
		info["logvar"] = logvar
		info["state"] = state_
		info["feats"] = feats
		return feats, info

	def vae_sampling(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)
	
	def torch_sampling(self, mu, log_var):
		z_dist = Normal(mu, torch.exp(0.5*log_var))
		z = z_dist.rsample()
		return z

	def decode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		"""
		ps. there is only one type of decoder since it is always from latent dim to the oracle obs
		"""
		info = {}
		encoder_outputs, state_ = self.decoder_net(input)
		res = encoder_outputs[0]
		return res, info


# Runner

"""new implementation"""
class DefaultRLRunner:
	""" Runner for RL algorithms
	Work flow: 
		entry.py
			initialize cfg.runner, pass cfg to runner
		runner.py(XXXRunner)
			initialize envs, policy, buffer, etc.
			call DefaultRLRunner.__init__(cfg) for mmon initialization
			call XXXRunner.__init__(cfg) for specific initialization
	"""
	def start(self, cfg):
		self.cfg = cfg
		self.env = cfg.env
		# init
		cfg.env.global_cfg = self.cfg.global_cfg
		seed = int(time()) if cfg.seed is None else cfg.seed
		utils.seed_everything(seed) # TODO add env seed
		self.env = utils.make_env(cfg.env) # to get obs_space and act_space
		self.console = Console()
		self.log = self.console.log
		self.log("Logger init done!")
		self.log("Checking cfg ...")
		self.check_cfg()
		self.log("_init_basic_components_for_all_alg ...")
		self._init_basic_components_for_all_alg()

	def _init_basic_components_for_all_alg(self):
		cfg = self.cfg
		env = self.env
		# basic for all algorithms
		self.global_cfg = cfg.global_cfg
		self.log("Init buffer & collector ...")
		self.buf = cfg.buffer
		self.train_collector = EnvCollector(env)
		self.test_collector = EnvCollector(env)
		self.log("Init others ...")
		self.record = WaybabaRecorder()
		self._start_time = time()
		if self.cfg.trainer.progress_bar:
			self.progress = Progress()
			self.progress.start()

	def check_cfg(self):
		print("You should implement check_cfg in your runner!")

class OfflineRLRunner(DefaultRLRunner):
	def start(self, cfg):
		super().start(cfg)
		self.log("init_components ...")
		self.init_components()

		self.log("_initial_exploration ...")
		self._initial_exploration()

		self.log("Training Start ...")
		self.env_step_global = 0
		if cfg.trainer.progress_bar: self.training_task = self.progress.add_task("[green]Training...", total=cfg.trainer.max_epoch*cfg.trainer.step_per_epoch)
		
		while True: # traininng loop
			# env step collect
			self._collect_once()
			
			# update
			if self._should_update(): 
				for _ in range(int(cfg.trainer.step_per_collect/cfg.trainer.update_per_step)):
					self.update_once()
			
			# evaluate
			if self._should_evaluate():
				self._evaluate()
			
			#log
			if self._should_write_log():
				self._log_time()
				self.record.upload_to_wandb(step=self.env_step_global, commit=False)
			
			# upload
			if self._should_upload_log():
				wandb.log({}, commit=True)
				
			# loop control
			if self._should_end(): break
			if cfg.trainer.progress_bar: self.progress.update(self.training_task, advance=cfg.trainer.step_per_collect, description=f"[green]🚀 Training {self.env_step_global}/{self.cfg.trainer.max_epoch*self.cfg.trainer.step_per_epoch}[/green]\n"+self.record.to_progress_bar_description())
			self.env_step_global += self.cfg.trainer.step_per_collect

		self._end_all()

	def init_components(self):
		raise NotImplementedError

	def _initial_exploration(self):
		"""exploration before training and add to self.buf"""
		initial_batches, info_ = self.train_collector.collect(
			act_func="random", n_step=self.cfg.start_timesteps, reset=True, 
			progress_bar="Initial Exploration ..." if self.cfg.trainer.progress_bar else None,
			rich_progress=self.progress if self.cfg.trainer.progress_bar else None
		)
		self.train_collector.reset()
		for batch in initial_batches: self.buf.add(batch)

	def _collect_once(self):
		"""collect data and add to self.buf"""

		batches, info_ = self.train_collector.collect(
			act_func=partial(self.select_act_for_env, mode="train"), 
			n_step=self.cfg.trainer.step_per_collect, reset=False
		)
		for batch in batches: self.buf.add(batch)

		# store history
		if not hasattr(self, "rew_sum_history"): self.rew_sum_history = []
		if not hasattr(self, "ep_len_history"): self.ep_len_history = []
		self.rew_sum_history += info_["rew_sum_list"]
		self.ep_len_history += info_["ep_len_list"]
		res_info = {
			"batches": batches,
			**info_
		}
		self._on_collect_end(**res_info)

	def _on_collect_end(self, **kwargs):
		"""called after a step of data collection"""
		self.on_collect_end(**kwargs)
	
	def update_once(self):
		raise NotImplementedError

	def _evaluate(self):
		"""Evaluate the performance of an agent in an environment.
		Args:
			env: Environment to evaluate on.
			act_func: Action selection function. It should take a single argument
				(observation) and return a single action.
		Returns:
			Episode reward.
		"""
		if not hasattr(self, "epoch_cnt"): self.epoch_cnt = 0
		# evaluate
		for mode in ["eval", "train"]:
			eval_type = "deterministic" if mode == "eval" else ""
			eval_batches, _ = self.test_collector.collect(
				act_func=partial(self.select_act_for_env, mode=mode), 
				n_episode=self.cfg.trainer.episode_per_test, reset=True,
				progress_bar=f"Evaluating {eval_type} ..." if self.cfg.trainer.progress_bar else None,
				rich_progress=self.progress if self.cfg.trainer.progress_bar else None,
			)
			eval_rews = [0. for _ in range(self.cfg.trainer.episode_per_test)]
			eval_lens = [0 for _ in range(self.cfg.trainer.episode_per_test)]
			cur_ep = 0
			for i, batch in enumerate(eval_batches): 
				eval_rews[cur_ep] += batch.rew
				eval_lens[cur_ep] += 1
				if batch.terminated or batch.truncated:
					cur_ep += 1
			self.record("eval/rew_mean"+"_"+eval_type, np.mean(eval_rews))
			self.record("eval/len_mean"+"_"+eval_type, np.mean(eval_lens))
		# loop control
		self.epoch_cnt += 1
		self.record("epoch", self.epoch_cnt)
		self._on_evaluate_end()
	
	def _on_evaluate_end(self):
		# print epoch log (deactivated in debug mode as the info is already in progress bar)
		if not self.cfg.trainer.hide_eval_info_print: 
			to_print = self.record.__str__().replace("\n", "  ")
			to_print = "[Epoch {: 5d}/{}] ### ".format(self.epoch_cnt-1, self.cfg.trainer.max_epoch) + to_print
			print(to_print)
		self.on_evaluate_end()
	
	def on_evaluate_end(self):
		pass

	def _log_time(self):
		if self.env_step_global > 1000:
			cur_time = time()
			hours_spent = (cur_time-self._start_time) / 3600
			speed = self.env_step_global / hours_spent
			hours_left = (self.cfg.trainer.max_epoch*self.cfg.trainer.step_per_epoch-self.env_step_global) / speed
			self.record("misc/hours_spent", hours_spent)
			self.record("misc/hours_left", hours_left)
			self.record("misc/step_per_hour", speed)

	def _end_all(self):
		if self.cfg.trainer.progress_bar: self.progress.stop()
		if self.cfg.env.save_minari: # save dataset
			version_ = self.cfg.trainer.max_epoch * self.cfg.trainer.step_per_epoch
			dataset_id = self.cfg.env.name.lower().split("-")[0]+f"-sac_{version_}"+"-v0"
			self.env.create_dataset(dataset_id=dataset_id)
			print("Minari dataset saved as name {}".format(dataset_id))
			# mv ~/.minari/datasets/{dataset_id} to os.environ['UDATADIR'] + /minari/datasets/{dataset_id}
			# if exist, replace
			import shutil
			source_dir = os.path.join(os.path.expanduser("~"), ".minari", "datasets", dataset_id)
			dest_dir = os.path.join(os.environ['UDATADIR'], "minari", "datasets")
			if os.path.exists(dest_dir+"/"+dataset_id): shutil.rmtree(dest_dir+"/"+dataset_id)
			if not os.path.exists(dest_dir): os.makedirs(dest_dir)
			shutil.move(source_dir, dest_dir)
			print("Dataset moved from {} to {}".format(source_dir, dest_dir))

	def select_act_for_env(self, batch, state, mode):
		"""
		Note this is only used when interacting with env. For learning state,
		the actions are got by calling self.actor ...
		Usage: 
			would be passed as a function to collector
		Args:
			batch: batch of data
			state: {"hidden": [], "hidden_pred": [], ...}
			mode: "train" or "eval"
				for train, it would use stochastic action
				for eval, it would use deterministic action
				usage: 
					1. when collecting data, mode="train" is used
					2. when evaluating, both mode="train" and mode="eval" are used
		"""
		raise NotImplementedError

	def _should_update(self):
		# TODO since we collect x steps, so we always update
		# if not hasattr(self, "should_update_record"): self.should_update_record = {}
		# cur_update_tick = self.env_step_global // self.cfg.trainer.
		# if cur_update_tick not in self.should_update_record:
		# 	self.should_update_record[cur_update_tick] = True
		# 	return True
		# return False
		return True
	
	def _should_evaluate(self):
		if not hasattr(self, "should_evaluate_record"): self.should_evaluate_record = {}
		cur_evaluate_tick = self.env_step_global // self.cfg.trainer.step_per_epoch
		if cur_evaluate_tick not in self.should_evaluate_record:
			self.should_evaluate_record[cur_evaluate_tick] = True
			return True
		return False
	
	def _should_write_log(self):
		if not hasattr(self, "should_log_record"): self.should_log_record = {}
		cur_log_tick = self.env_step_global // self.cfg.trainer.log_interval
		if cur_log_tick not in self.should_log_record:
			self.should_log_record[cur_log_tick] = True
			return True
		return False

	def _should_upload_log(self):
		if not self.cfg.trainer.log_upload_interval: return True # missing or zero, always upload
		if not hasattr(self, "should_upload_record"): self.should_upload_record = {}
		cur_upload_tick = self.env_step_global // self.cfg.trainer.log_upload_interval
		if cur_upload_tick not in self.should_upload_record:
			self.should_upload_record[cur_upload_tick] = True
			return True
		return False

	def _should_end(self):
		return self.env_step_global >= self.cfg.trainer.max_epoch * self.cfg.trainer.step_per_epoch

class TD3SACRunner(OfflineRLRunner):
	def check_cfg(self):
		cfg = self.cfg
		global_cfg = cfg.global_cfg
		if global_cfg.critic_input.history_merge_method == "stack_rnn":
			pass

	def init_components(self):
		self.log("Init networks ...")
		env = self.env
		cfg = self.cfg

		# networks
		self.actor = cfg.actor(state_shape=env.observation_space.shape, action_shape=env.action_space.shape, max_action=env.action_space.high[0],global_cfg=self.cfg.global_cfg).to(cfg.device)
		self.actor_optim = cfg.actor_optim(self.actor.parameters())
		self.actor_old = deepcopy(self.actor)
		
		# decide bi or two direction
		if cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			if cfg.global_cfg.critic_input.bi_or_si_rnn == "si":
				self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
			elif cfg.global_cfg.critic_input.bi_or_si_rnn == "bi":
				self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
			elif cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
				self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic_sirnn_1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic_sirnn_2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic_sirnn_1_optim = cfg.critic1_optim(self.critic_sirnn_1.parameters())
				self.critic_sirnn_2_optim = cfg.critic2_optim(self.critic_sirnn_2.parameters())
			else:
				raise NotImplementedError
		else:
			self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
			self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
		self.critic1_optim = cfg.critic1_optim(self.critic1.parameters())
		self.critic2_optim = cfg.critic2_optim(self.critic2.parameters())
		self.critic1_old = deepcopy(self.critic1)
		self.critic2_old = deepcopy(self.critic2)

		self.actor_old.eval()
		self.critic1.train()
		self.critic2.train()
		self.critic1_old.train()
		self.critic2_old.train()
		
		if self.ALGORITHM == "td3":
			redudent_net = []
			self.exploration_noise = cfg.policy.initial_exploration_noise
			# TODO remove dedudent net
		if self.ALGORITHM == "sac":
			redudent_net = ["actor_old"]
			self.critic1_old.eval()
			self.critic2_old.eval()
			self.log("init sac alpha ...")
			self._init_sac_alpha()
		if self.ALGORITHM == "ddpg":
			redudent_net = ["critic2", "critic2_old"]
			self.actor_old = deepcopy(self.actor)
			self.actor_old.eval()
			self.exploration_noise = cfg.policy.initial_exploration_noise
		
		if redudent_net:
			for net_name in redudent_net: delattr(self, net_name)
		# obs pred & encode
		assert not (self.global_cfg.actor_input.obs_pred.turn_on and self.global_cfg.actor_input.obs_encode.turn_on), "obs_pred and obs_encode cannot be used at the same time"
		
		if self.global_cfg.actor_input.obs_pred.turn_on:
			self.pred_net = self.global_cfg.actor_input.obs_pred.net(
				state_shape=self.env.observation_space.shape,
				action_shape=self.env.action_space.shape,
				global_cfg=self.global_cfg,
			)
			self._pred_optim = self.global_cfg.actor_input.obs_pred.optim(
				self.pred_net.parameters(),
			)
			if self.global_cfg.actor_input.obs_pred.auto_kl_target:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				)], device=self.actor.device, requires_grad=True)
				self._auto_kl_optim = self.global_cfg.actor_input.obs_pred.auto_kl_optim([self.kl_weight_log])
			else:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				)], device=self.actor.device)
		
		if self.global_cfg.actor_input.obs_encode.turn_on:
			self.encode_net = self.global_cfg.actor_input.obs_encode.net(
				state_shape=self.env.observation_space.shape,
				action_shape=self.env.action_space.shape,
				global_cfg=self.global_cfg,
			)
			self._encode_optim = self.global_cfg.actor_input.obs_encode.optim(
				self.encode_net.parameters(),
			)
			if self.global_cfg.actor_input.obs_encode.auto_kl_target:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_encode.norm_kl_loss_weight
				)], device=self.actor.device, requires_grad=True)
				self._auto_kl_optim = self.global_cfg.actor_input.obs_encode.auto_kl_optim([self.kl_weight_log])
			else:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_encode.norm_kl_loss_weight
				)], device=self.actor.device)
	
	def preprocess_from_env(self, batch, state, mode=None):
		"""
		use in the online interaction with env
		"""
		assert len(batch.obs.shape) == 1, "for online batch, batch size must be 1"
		res_state = {}

		# add "hidden" since it is not used in this function
		res_state = update_state(res_state, state, {"hidden": "hidden"})

		if self.global_cfg.actor_input.obs_type == "normal": a_in = batch.obs
		elif self.global_cfg.actor_input.obs_type == "oracle": a_in = batch.info["obs_next_nodelay"]

		if self.global_cfg.actor_input.history_merge_method == "none":
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_out, pred_info = self.pred_net(a_in)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_out.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else: raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				pred_abs_error_online = ((pred_out - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
				self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				encode_output, encode_info = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					pred_obs_output_cur, _ = self.encode_net.decode(encode_output)
					pred_abs_error_online = ((pred_obs_output_cur - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
					self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
		
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if self.global_cfg.history_num > 0: # only cat when > 0
				a_in = np.concatenate([
					a_in,
					batch.info["historical_act"].flatten() if not self.cfg.global_cfg.debug.new_his_act \
					else batch.info["historical_act_next"].flatten()
			], axis=-1)
			
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_out, pred_info = self.pred_net(a_in)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_out.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else: raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				
				pred_abs_error_online = ((pred_out - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
				self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
			
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				encode_output, encode_info = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					pred_obs_output_cur, _ = self.encode_net.decode(encode_output)
					pred_abs_error_online = ((pred_obs_output_cur - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
					self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
		
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			if self.global_cfg.history_num > 0:
				latest_act = batch.info["historical_act"][-1] if not self.cfg.global_cfg.debug.new_his_act \
				else batch.info["historical_act_next"][-1]
				a_in = np.concatenate([a_in, latest_act], axis=-1)
			if self.global_cfg.actor_input.obs_pred.turn_on:
				state_for_obs_pred = distill_state(state, {"hidden_pred_net_encoder": "hidden_encoder", "hidden_pred_net_decoder": "hidden_decoder"})
				pred_out, pred_info = self.pred_net(a_in, state=state_for_obs_pred)
				res_state = update_state(res_state, state_for_obs_pred, {"hidden_encoder": "hidden_pred_net_encoder", "hidden_decoder": "hidden_pred_net_decoder"})
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_out.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else: raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				if "is_first_step" not in batch.info:
					pred_abs_error_online = ((pred_out - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
					self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError("stack_rnn + obs_encode not implemented")
				encode_output, res_state = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()

		elif self.global_cfg.actor_input.history_merge_method == "transformer":
			if self.global_cfg.history_num > 0:
				a_in = np.concatenate([
					batch.info["historical_obs_next"], # TODO here we does not consider the oracle obs
					batch.info["historical_act_next"],
				], axis=-1)
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pass # TODO transformer
				raise NotImplementedError("transformer + obs_pred not implemented")
				# state_for_obs_pred = distill_state(state, {"hidden_pred_net_encoder": "hidden_encoder", "hidden_pred_net_decoder": "hidden_decoder"})
				# pred_out, pred_info = self.pred_net(a_in, state=state_for_obs_pred)
				# res_state = update_state(res_state, state_for_obs_pred, {"hidden_encoder": "hidden_pred_net_encoder", "hidden_decoder": "hidden_pred_net_decoder"})
				# if self.global_cfg.actor_input.obs_pred.input_type == "obs":
				# 	a_in = pred_out.cpu()
				# elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
				# 	a_in = pred_info["feats"].cpu()
				# else: raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				# if "is_first_step" not in batch.info:
				# 	pred_abs_error_online = ((pred_out - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
				# 	self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				pass # TODO transformer
				raise NotImplementedError("transformer + obs_encode not implemented")
				# raise NotImplementedError("stack_rnn + obs_encode not implemented")
				# encode_output, res_state = self.encode_net.normal_encode(a_in)
				# a_in = encode_output.cpu()
		
		else:
			raise ValueError(f"history_merge_method {self.global_cfg.actor_input.history_merge_method} not implemented")
		
		return a_in, res_state
	
	def select_act_for_env(self, batch, state, mode):
		a_in, res_state = self.preprocess_from_env(batch, state, mode=mode)
		
		# forward
		if not isinstance(a_in, torch.Tensor): a_in = torch.tensor(a_in, dtype=torch.float32).to(self.cfg.device)
		if self.cfg.global_cfg.debug.abort_infer_state:
			state_for_a = None
		else:
			state_for_a = distill_state(state, {"hidden": "hidden"})
		a_out, actor_state = self.actor(a_in, state_for_a)
		res_state = update_state(res_state, actor_state, {"hidden": "hidden"})
		
		if self.ALGORITHM == "td3":
			if mode == "train":
				a_out = a_out[0]
				# noise = torch.tensor(self._noise(a_out.shape), device=self.cfg.device)
				a_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
				noise = torch.normal(0., a_scale * self.exploration_noise).to(device=self.cfg.device)
				res = a_out + noise
				# if self.cfg.policy.noise_clip > 0.0:
				# 	noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
				res = res.clip(
					torch.tensor(self.env.action_space.low, device=self.cfg.device),
					torch.tensor(self.env.action_space.high, device=self.cfg.device),
				)
			elif mode == "eval":
				res = a_out[0]
			else: 
				raise ValueError("unknown mode: {}".format(mode))
		elif self.ALGORITHM == "sac":
			if mode == "train":
				assert isinstance(a_out, tuple) # (mean, logvar)
				dist = Independent(Normal(*a_out), 1)
				act = dist.rsample()
				squashed_action = torch.tanh(act) * torch.tensor(self.env.action_space.high, device=self.cfg.device) + 0.0 # TODO bias
				# log_prob = dist.log_prob(act).unsqueeze(-1)
				# log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
				# 								np.finfo(np.float32).eps.item()).sum(-1, keepdim=True) # TODO remove, seems not used 
				res = squashed_action
			elif mode == "eval":
				res = a_out[0]
				res = torch.tanh(res) * torch.tensor(self.env.action_space.high, device=self.cfg.device) + 0.0 # TODO bias
			else: raise ValueError("unknown mode: {}".format(mode))
		elif self.ALGORITHM == "ddpg":
			if mode == "train":
				a_out = a_out[0]
				# noise = torch.tensor(self._noise(a_out.shape), device=self.cfg.device)
				a_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
				noise = torch.normal(0., a_scale * self.exploration_noise).to(device=self.cfg.device)
				res = a_out + noise
				# if self.cfg.policy.noise_clip > 0.0:
				# 	noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
				res = res.clip(
					torch.tensor(self.env.action_space.low, device=self.cfg.device),
					torch.tensor(self.env.action_space.high, device=self.cfg.device),
				)
			elif mode == "eval":
				res = a_out[0]
			else: 
				raise ValueError("unknown mode: {}".format(mode))

		return res, res_state if res_state else None
	
	def on_collect_end(self, **kwargs):
		"""called after a step of data collection"""
		if "rew_sum_list" in kwargs and kwargs["rew_sum_list"]:
			for i in range(len(kwargs["rew_sum_list"])): self.record("collect/rew_sum", kwargs["rew_sum_list"][i])
		if "ep_len_list" in kwargs and kwargs["ep_len_list"]:
			for i in range(len(kwargs["ep_len_list"])): self.record("collect/ep_len", kwargs["ep_len_list"][i])

	def update_once(self):
		# indices = self.buf.sample_indices(self.cfg.trainer.batch_size)
		idxes, idxes_remaster, valid_mask = self.buf.sample_indices_remaster(self.cfg.trainer.batch_size, self.cfg.trainer.batch_seq_len)
		batch = self._indices_to_batch(idxes)
		batch.valid_mask = valid_mask
		if self._burnin_num(): # TODO move all of this to buffer would make the logic nicer
			burnin_idxes, _, burnin_mask = self.buf.create_burnin_pair(idxes_remaster, self._burnin_num())
			burnin_batch = self._indices_to_batch(burnin_idxes)
			burnin_batch.valid_mask = burnin_mask
			batch = self._pre_update_process(batch, burnin_batch) # would become state in batch.state
		else:
			batch = self._pre_update_process(batch)

		if not hasattr(self, "critic_update_cnt"): self.update_cnt = 0
		
		if self.ALGORITHM == "td3":
			# update cirtic
			self.update_critic(batch)
			if self.update_cnt % self.cfg.policy.update_a_per_c == 0:
				# update actor
				self.update_actor(batch)
				self.exploration_noise *= self.cfg.policy.noise_decay_rate
			self._soft_update(self.actor_old, self.actor, self.cfg.policy.tau)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.critic2_old, self.critic2, self.cfg.policy.tau)
		elif self.ALGORITHM == "sac":
			self.update_critic(batch)
			self.update_actor(batch)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.critic2_old, self.critic2, self.cfg.policy.tau)
		elif self.ALGORITHM == "ddpg":
			self.update_critic(batch)
			self.update_actor(batch)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.actor_old, self.actor, self.cfg.policy.tau)
		else:
			raise NotImplementedError
			
		self.update_cnt += 1
	
	def update_critic(self, batch):
		raise NotImplementedError
	
	def update_actor(self, batch):
		raise NotImplementedError

	def _indices_to_batch(self, indices):
		""" sample batch from buffer with indices
		After sampling, indices are discarded. So we need to make sure that
		all the information we need is in the batch.

		TODO souce of historical act, now is from info["histo...], can also from act

		Exaplanation of the batch keys:
			` basic ones from the buffer: obs, act, obs_next, rew, done, terminated, truncated, policy`
			dobs, dobs_next (*, obs_dim): delayed obs, delayed obs_next, inherited from obs, obs_next. To
				avoid confusion, the obs, obs_next would be renamed to these two names.
			oobs, oobs_next (*, obs_dim): oracle obs, oracle obs_next
			act, rew, done (*, ): no changes (terminated and truncated are removed since we don't need them)
			ahis_cur, ahis_next (*, history_len, act_dim): history of actions
		"""
		keeped_keys = ["dobs", "dobs_next", "oobs", "oobs_next", "ahis_cur", "ahis_next", "ohis_cur","ohis_next", "act", "rew", "done", "terminated", "obs_delayed_step_num"]
		batch = self.buf[indices]
		batch.dobs, batch.dobs_next = batch.obs, batch.obs_next
		batch.oobs, batch.oobs_next = batch.info["obs_nodelay"], batch.info["obs_next_nodelay"]
		
		if self.cfg.global_cfg.debug.new_his_act:
			batch.ahis_cur = batch.info["historical_act_cur"]
			batch.ahis_next = batch.info["historical_act_next"]
			batch.ohis_cur = batch.info["historical_obs_cur"]
			batch.ohis_next = batch.info["historical_obs_next"]
		else:
			batch.ahis_cur = batch.info["historical_act"]
			batch.ahis_next = self.buf[self.buf.next(indices)].info["historical_act"]
		
		batch.obs_delayed_step_num = batch.info["obs_delayed_step_num"]
		for k in list(batch.keys()): 
			if k not in keeped_keys:
				batch.pop(k)
		return batch

	def _pre_update_process(self, batch, burnin_batch=None):
		""" Pre-update process
		including merging history, obs_pred, obs_encode ... and removing some keys 
		only keep keys used in update
		input keys: ["dobs", "dobs_next", "oobs", "oobs_next", "ahis_cur", "ahis_next", "act", "rew", "done"]
		output keys:
			"a_in_cur", "a_in_next", 
			"c_in_online_cur", "c_in_online_next", "c_in_cur",
			"done", "rew", "act", "valid_mask", "terminated"
			(only when obs_pred) "pred_out_cur", "oobs", 
			(only when obs_encode)
			(only when sac) "logprob_online_cur", "logprob_online_next"
			ps. the key with "online" is with gradient
		"""
		keeped_keys = ["a_in_cur", "a_in_next", "c_in_cur", "c_in_online_cur", "c_in_online_next", "logprob_online_cur", "logprob_online_next", "done", "rew", "act", "valid_mask", "terminated"]
		batch.to_torch(device=self.cfg.device, dtype=torch.float32)
		if self._burnin_num(): burnin_batch.to_torch(device=self.cfg.device, dtype=torch.float32)
		pre_sz = list(batch["done"].shape)
		

		# actor - 1. obs base
		if self.global_cfg.actor_input.obs_type == "normal": 
			batch.a_in_cur = batch.dobs
			batch.a_in_next = batch.dobs_next
			if self._burnin_num(): burnin_batch.a_in = burnin_batch.dobs # only need a_in since we dont dont forward twice, the last state of cur would be used in next
		elif self.global_cfg.actor_input.obs_type == "oracle": 
			batch.a_in_cur  = batch.oobs
			batch.a_in_next = batch.oobs_next
			if self._burnin_num(): burnin_batch.a_in  = burnin_batch.oobs
		else:
			raise ValueError("unknown obs_type: {}".format(self.global_cfg.actor_input.obs_type))

		# actor - 2. others
		if self.global_cfg.actor_input.history_merge_method == "none":
			# TODO seems that the obs_pred and obs_encode can be outside
			if self.global_cfg.actor_input.obs_pred.turn_on:
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()	
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if self.global_cfg.history_num > 0: 
				batch.a_in_cur = torch.cat([batch.a_in_cur, batch.ahis_cur.flatten(start_dim=-2)], dim=-1)
				batch.a_in_next = torch.cat([batch.a_in_next, batch.ahis_next.flatten(start_dim=-2)], dim=-1)
			
			if self.global_cfg.actor_input.obs_pred.turn_on:
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()	
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			if self.global_cfg.history_num > 0: 
				batch.a_in_cur = torch.cat([batch.a_in_cur, batch.ahis_cur[...,-1,:]], dim=-1) # [*, obs_dim+act_dim]
				batch.a_in_next = torch.cat([batch.a_in_next, batch.ahis_next[...,-1,:]], dim=-1)
				if self._burnin_num(): 
					burnin_batch.a_in = torch.cat([burnin_batch.a_in, burnin_batch.ahis_cur[...,-1,:]], dim=-1) # [B, T, obs_dim+act_dim]
			
			if self._burnin_num():
				keeped_keys += ["burnin_a_in", "burnin_remaster_mask"] # mask reused by critic
				batch.burnin_a_in, batch.burnin_remaster_mask = burnin_batch.a_in, burnin_batch.valid_mask

			if self.global_cfg.actor_input.obs_pred.turn_on:
				raise NotImplementedError
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
		elif self.global_cfg.actor_input.history_merge_method == "transformer":
			if self.global_cfg.history_num > 0: 
				batch.a_in_cur = torch.cat([batch.ohis_cur, batch.ahis_cur], dim=-1) # [*, obs_dim+act_dim]
				batch.a_in_next = torch.cat([batch.ohis_next, batch.ahis_next], dim=-1)

			if self.global_cfg.actor_input.obs_pred.turn_on:
				raise NotImplementedError
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
		
		else: 
			raise ValueError("history_merge_method error")

		# get online act from actor
		act_online, act_online_next, act_online_info = self.get_act_online(batch)
		if "logprob_online_cur" in act_online_info: batch.logprob_online_cur = act_online_info["logprob_online_cur"]
		if "logprob_online_next" in act_online_info: batch.logprob_online_next = act_online_info["logprob_online_next"]

		# critic - 1. obs base
		if self.global_cfg.critic_input.obs_type == "normal": 
			batch.c_in_cur = batch.dobs
			batch.c_in_next = batch.dobs_next
			if self._burnin_num(): burnin_batch.c_in  = burnin_batch.dobs
		elif self.global_cfg.critic_input.obs_type == "oracle": 
			batch.c_in_cur  = batch.oobs
			batch.c_in_next = batch.oobs_next
			if self._burnin_num(): burnin_batch.c_in  = burnin_batch.oobs
		else:
			raise ValueError("unknown obs_type: {}".format(self.global_cfg.critic_input.obs_type))

		# critic - 2. merge act
		batch.c_in_online_cur = torch.cat([batch.c_in_cur, act_online], dim=-1)
		batch.c_in_online_next = torch.cat([batch.c_in_next, act_online_next], dim=-1)
		batch.c_in_cur = torch.cat([batch.c_in_cur, batch.act], dim=-1)
		if self._burnin_num(): burnin_batch.c_in = torch.cat([burnin_batch.c_in, burnin_batch.act], dim=-1) # [B, T, obs_dim+act_dim]

		# critic - 3. merge act history
		if self.cfg.global_cfg.critic_input.history_merge_method == "none":
			pass
		elif self.cfg.global_cfg.critic_input.history_merge_method == "cat_mlp":
			if self.global_cfg.history_num > 0:
				batch.c_in_online_cur = torch.cat([batch.c_in_online_cur, batch.ahis_cur.flatten(start_dim=-2)], dim=-1)
				batch.c_in_online_next = torch.cat([batch.c_in_online_next, batch.ahis_next.flatten(start_dim=-2)], dim=-1)
				batch.c_in_cur = torch.cat([batch.c_in_cur, batch.ahis_cur.flatten(start_dim=-2)], dim=-1)
		elif self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			# TODO the state for critic varies a lot across critic1, critic2, criticold, so we cannot save the state then process
			assert self.global_cfg.history_num == 1, "in stack_rnn, history_num must be 1"
			assert 1, "RNN layer must be 1 since in current implementation we can only get state of last layer"
			keeped_keys += ["preinput"]
			batch.c_in_online_cur = torch.cat([batch.c_in_online_cur, batch.ahis_cur[...,-1,:]], dim=-1)
			batch.c_in_online_next = torch.cat([batch.c_in_online_next, batch.ahis_next[...,-1,:]], dim=-1)
			batch.c_in_cur = torch.cat([batch.c_in_cur, batch.ahis_cur[...,-1,:]], dim=-1)
			batch.preinput = batch.c_in_cur
			if self._burnin_num(): # TODO
				burnin_batch.c_in = torch.cat([burnin_batch.c_in, burnin_batch.ahis_cur[...,-1,:]], dim=-1) # [B, T, obs_dim+act_dim+act_dim*act_dim]
			if self._burnin_num():
				keeped_keys += ["burnin_c_in", "burnin_remaster_mask"] # mask reused by critic
				batch.burnin_c_in, batch.burnin_remaster_mask = burnin_batch.c_in, burnin_batch.valid_mask
		else:
			raise ValueError("unknown history_merge_method: {}".format(self.cfg.global_cfg.critic_input.history_merge_method))
		
		# log
		self.record("learn/obs_delayed_step_num", batch.obs_delayed_step_num.mean().item())
		self.record("learn/obs_delayed_step_num_sample", batch.obs_delayed_step_num.flatten()[0].item())

		# only keep res keys
		for k in list(batch.keys()): 
			if k not in keeped_keys: batch.pop(k)
		
		return batch

	def on_evaluate_end(self):
		"""called after a step of evaluation"""
		pass

	def _get_historical_act(self, indices, step, buffer, type=None, device=None):
		""" get historical act
		input [t_0, t_1, ...]
		output [
			[t_0-step, t_0-step+1, ... t_0-1],
			[t_1-step, t_1-step+1, ... t_1-1],
			...
		]
		ps. note that cur step is not included
		ps. the neg step is set to 0.
		:param indices: indices of the batch (B,)
		:param step: the step of the batch. int
		:param buffer: the buffer. 
		:return: historical act (B, step)
		"""
		raise ValueError("Deprecated")
		assert type in ["cat", "stack"], "type must be cat or stack"
		# [t_0-step, t_0-step+1, ... t_0-1, t_0]
		idx_stack_plus1 = utils.idx_stack(indices, buffer, step+1, direction="prev")
		# [t_0-step,   t_0-step+1, ..., t_0-1]
		idx_stack_next = idx_stack_plus1[:, :-1] # (B, step)
		# [t_0-step+1, t_0-step+2, ...,   t_0]
		idx_stack = idx_stack_plus1[:, 1:] # (B, step)
		invalid = (idx_stack_next == idx_stack) # (B, step)
		historical_act = buffer[idx_stack].act # (B, step, act_dim)
		historical_act[invalid] = 0.
		if type == "cat":
			historical_act = historical_act.reshape(historical_act.shape[0], -1) # (B, step*act_dim)
		historical_act = torch.tensor(historical_act, device=device)
		return historical_act

	def adjust_idx_stack(self, idx_stack, adjust_dis, buffer):
		"""
		:param idx_stack: (B, T)
		:param adjust_dis: int
		:param buffer: the buffer
		if the idx_stack start is < adjust_dis to the start of the buffer, then adjust it to the start
		"""
		idx_start = idx_stack[:, 0]
		for _ in range(adjust_dis):
			idx_start = buffer.prev(idx_start)
		idx_to_adjust = idx_start == buffer.prev(idx_start)
		idx_start_to_stack_list = []
		idx_start_to_stack = idx_start.copy()
		for i in range(idx_stack.shape[1]):
			idx_start_to_stack_list.append(idx_start_to_stack)
			idx_start_to_stack = buffer.next(idx_start_to_stack)
		idx_stack_all_adjusted = np.stack(idx_start_to_stack_list, axis=1)
		idx_stack[idx_to_adjust] = idx_stack_all_adjusted[idx_to_adjust]
		return idx_stack

	def get_historical_act(self, indices, step, buffer, type=None, device=None):
		# Get the action dimension from the buffer
		act_dim = buffer[0].act.shape[0]

		# Determine the output shape based on the type
		if type == "stack":
			output_shape = (*indices.shape, step, act_dim)
		elif type == "cat":
			output_shape = (*indices.shape, step * act_dim)
		else:
			raise ValueError("Invalid type: choose 'stack' or 'cat'")

		# Create an empty tensor with the output shape
		res = np.zeros(output_shape)

		# Iterate through the input indices and retrieve previous actions
		for i in range(step - 1, -1, -1):  # Reverse loop using a single line
			prev_indices = buffer.prev(indices)
			idx_start = prev_indices == indices
			res_batch_act = buffer[prev_indices].act

			# Handle the case when the requested action is at the start of the buffer
			res_batch_act[idx_start] = 0.

			# Fill the output tensor with the retrieved actions
			if type == "stack":
				res[..., i, :] = res_batch_act
			elif type == "cat":
				res[..., i * act_dim:(i + 1) * act_dim] = res_batch_act

			indices = prev_indices
		
		# Convert the output tensor to a torch tensor
		res = torch.tensor(res, device=device)

		return res

	def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
		"""Softly update the parameters of target module towards the parameters \
		of source module."""
		for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
			tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
				# update the target network

	def _burnin_num(self):
		if "burnin_num" not in self.cfg.global_cfg: return 0
		if not self.cfg.global_cfg.burnin_num: return 0
		if self.cfg.global_cfg.actor_input.history_merge_method != "stack_rnn": return 0
		burnin_num = self.cfg.global_cfg.burnin_num
		if type(self.cfg.global_cfg.burnin_num) == float:
			burnin_num = int(self.cfg.global_cfg.burnin_num * self.cfg.trainer.batch_seq_len)
		elif type(self.cfg.global_cfg.burnin_num) == int:
			burnin_num = self.cfg.global_cfg.burnin_num
		return burnin_num

	def get_act_online(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Returns the current and next actions to be taken by the agent.

		ps. include all actor forwards during learning

		Motivation: the forward of the actor is across algorithms,
			more importantly, the output results should be used to make 
			critic input

		Returns:
			A tuple of two torch.Tensor objects representing the current and next actions.
		"""
		raise NotImplementedError

	def add_obs_pred_loss(self, batch, combined_loss):
		pred_cfg = self.global_cfg.actor_input.obs_pred
		pred_loss = (batch.pred_out_cur - batch.oobs) ** 2
		pred_loss = apply_mask(pred_loss, batch.valid_mask).mean()
		pred_loss_normed = pred_loss / batch.valid_mask.float().mean()
		combined_loss += pred_loss.mean() * pred_cfg.pred_loss_weight
		self.record("learn/obs_pred/pred_loss", pred_loss.item())
		self.record("learn/obs_pred/pred_loss_normed", pred_loss_normed.item())
		self.record("learn/obs_pred/pred_abs_error", pred_loss.item() ** 0.5)
		self.record("learn/obs_pred/pred_abs_error_normed", pred_loss_normed.item() ** 0.5)
		if pred_cfg.net_type == "vae":
			kl_loss = kl_divergence(
				batch.pred_info_cur_mu,
				batch.pred_info_cur_logvar,
				torch.zeros_like(batch.pred_info_cur_mu),
				torch.zeros_like(batch.pred_info_cur_logvar),
			)
			kl_loss = apply_mask(kl_loss, batch.valid_mask).mean()
			kl_loss_normed = kl_loss / batch.valid_mask.float().mean()
			combined_loss += kl_loss * torch.exp(self.kl_weight_log).detach()
			self.record("learn/obs_pred/loss_kl", kl_loss.item())
			self.record("learn/obs_pred/loss_kl_normed", kl_loss_normed.item())
			if pred_cfg.auto_kl_target:
				kl_weight_loss = - (kl_loss_normed.detach() - pred_cfg.auto_kl_target) * torch.exp(self.kl_weight_log)
				self._auto_kl_optim.zero_grad()
				kl_weight_loss.backward()
				self._auto_kl_optim.step()
				self.record("learn/obs_pred/kl_weight_log", self.kl_weight_log.detach().cpu().item())
				self.record("learn/obs_pred/kl_weight", torch.exp(self.kl_weight_log).detach().cpu().item())
		return combined_loss
	
	def add_obs_encode_loss(self, batch, combined_loss):
		encode_cfg = self.global_cfg.actor_input.obs_encode
		kl_loss = kl_divergence(batch.encode_oracle_info_cur_mu, batch.encode_oracle_info_cur_logvar, batch.encode_normal_info_cur_mu, batch.encode_normal_info_cur_logvar)
		kl_loss = apply_mask(kl_loss, batch.valid_mask).mean()
		kl_loss_normed = kl_loss / batch.valid_mask.float().mean()
		combined_loss += kl_loss * torch.exp(self.kl_weight_log).detach().mean()
		self.record("learn/obs_encode/loss_kl", kl_loss.item())
		self.record("learn/obs_encode/loss_kl_normed", kl_loss_normed.item())

		if encode_cfg.pred_loss_weight:
			pred_loss = (batch.pred_obs_output_cur - batch.oobs) ** 2
			pred_loss = apply_mask(pred_loss, batch.valid_mask).mean()
			self.record("learn/obs_encode/loss_pred", pred_loss.item())
			self.record("learn/obs_encode/abs_error_pred", pred_loss.item() ** 0.5)
			combined_loss += pred_loss * encode_cfg.pred_loss_weight
		
		if encode_cfg.policy_robust_weight:
			dist = Normal(
				batch.encode_normal_info_cur_mu, 
				torch.exp(0.5*batch.encode_normal_info_cur_logvar)
			)
			z_1, z_2 = dist.sample(), dist.sample()
			(a_mu_1, a_var_1), _ = self.actor(z_1, None)
			(a_mu_2, a_var_2), _ = self.actor(z_2, None)
			robust_loss = (a_mu_1 - a_mu_2) ** 2 + (a_var_1.sqrt() - a_var_2.sqrt()) ** 2
			robust_loss = apply_mask(robust_loss, batch.valid_mask).mean()
			robust_loss_normed = robust_loss / batch.valid_mask.float().mean()
			self.record("learn/obs_encode/loss_robust", robust_loss.item())
			self.record("learn/obs_encode/loss_robust_normed", robust_loss_normed.item())
			combined_loss += robust_loss * encode_cfg.policy_robust_weight

		if encode_cfg.auto_kl_target:
			if self.global_cfg.debug.auto_kl_use_log:  # in paper
				kl_weight_loss = - self.kl_weight_log * (
					torch.log10(torch.clamp(kl_loss_normed.detach(), 1e-9, np.inf)) - \
					np.log10(encode_cfg.auto_kl_target)
				)
			else: # previous
				kl_weight_loss = - torch.exp(self.kl_weight_log) * (
					kl_loss_normed.detach() - \
					encode_cfg.auto_kl_target
				) 
			
			if self.global_cfg.debug.auto_kl_divide_act_dim: # in paper
				kl_weight_loss = kl_weight_loss / self.actor.act_num
			
			self._auto_kl_optim.zero_grad()
			kl_weight_loss.backward()
			self._auto_kl_optim.step()
			self.record("learn/obs_encode/kl_weight_log", self.kl_weight_log.detach().cpu().item())
			self.record("learn/obs_encode/kl_weight", torch.exp(self.kl_weight_log).detach().cpu().item())
		return combined_loss
		
# algorithms

class TD3Runner(TD3SACRunner):
	"""
	TODO the sirnn for TD3 and DDPG is not implemented yet
	"""
	ALGORITHM = "td3"

	def update_critic(self, batch):
		pre_sz = list(batch.done.shape)

		if self.cfg.global_cfg.debug.use_terminated_mask_for_value:
			value_mask = batch.terminated
		else:
			value_mask = batch.done

		with torch.no_grad():
			target_q = (batch.rew + self.cfg.policy.gamma * (1 - value_mask.int()) * \
				torch.min(
					self.critic1_old(batch.c_in_online_next, None)[0],
					self.critic2_old(batch.c_in_online_next, None)[0]
				).squeeze(-1)
			).reshape(*pre_sz,-1)
		
		critic_loss = \
			F.mse_loss(self.critic1(
				batch.c_in_cur, None
			)[0].reshape(*pre_sz,-1), target_q, reduce=False) + \
			F.mse_loss(self.critic2(
				batch.c_in_cur, None
			)[0].reshape(*pre_sz,-1), target_q, reduce=False)
			
		critic_loss = apply_mask(critic_loss, batch.valid_mask).mean()
		self.record("learn/critic_loss", critic_loss.cpu().item())
		self.record("learn/valid_mask_ratio", batch.valid_mask.float().mean())
		
		self.critic1_optim.zero_grad()
		self.critic2_optim.zero_grad()
		critic_loss.backward()
		self.critic1_optim.step()
		self.critic2_optim.step()

	def update_actor(self, batch):
		res_info = {}
		actor_loss, _ = self.critic1(batch.c_in_online_cur, None)
		actor_loss =  - apply_mask(actor_loss, batch.valid_mask).mean()
		combined_loss = 0. + actor_loss

		# add obs_pred loss
		if self.global_cfg.actor_input.obs_pred.turn_on:
			combined_loss = self.add_obs_pred_loss(batch, combined_loss)
		
		# add obs_encode loss
		if self.global_cfg.actor_input.obs_encode.turn_on:
			combined_loss = self.add_obs_encode_loss(batch, combined_loss)

		# backward and optim
		self.actor_optim.zero_grad()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.zero_grad()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.zero_grad()
		combined_loss.backward()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.step()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.step()
		self.actor_optim.step()
		self.record("learn/actor_loss", actor_loss.cpu().item())

	def get_act_online(self, batch):
		act_online_cur = self.actor(
			batch.a_in_cur, 
			state=batch.a_state_cur if self._burnin_num() else None
		)[0][0]
		with torch.no_grad():
			a_next_online_old = self.actor_old(
				batch.a_in_next, 
				state=batch.a_state_next if self._burnin_num() else None
			)[0][0]
			noise = torch.randn(size=a_next_online_old.shape, device=a_next_online_old.device)
			noise *= self.cfg.policy.policy_noise
			if self.cfg.policy.noise_clip > 0.0:
				noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
				noise *= torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2, device=a_next_online_old.device)
			a_next_online_old += noise
			a_next_online_old = a_next_online_old.clamp(self.env.action_space.low[0], self.env.action_space.high[0])
			act_online_next = a_next_online_old
			
		return act_online_cur, act_online_next, {}
	
class SACRunner(TD3SACRunner):
	ALGORITHM = "sac"

	def update_critic(self, batch):
		# cal target_q
		pre_sz = list(batch.done.shape)
		
		if self.cfg.global_cfg.debug.use_terminated_mask_for_value:
			value_mask = batch.terminated
		else:
			value_mask = batch.done
		
		# target
		with torch.no_grad():
			if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
				v_next_1 = forward_with_preinput(batch.c_in_online_next, batch.preinput, self.critic1_old, "next")
				v_next_2 = forward_with_preinput(batch.c_in_online_next, batch.preinput, self.critic2_old, "next")
			else:
				v_next_1 = self.critic1_old(batch.c_in_online_next, None)[0]
				v_next_2 = self.critic2_old(batch.c_in_online_next, None)[0]
			v_next = torch.min(
				v_next_1, v_next_2
			).reshape(*pre_sz) - self._log_alpha.exp().detach() * batch.logprob_online_next.reshape(*pre_sz)
			target_q = batch.rew + self.cfg.policy.gamma * (1 - value_mask.int()) * v_next
		
		# cur
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			v_cur_1 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic1, "cur")
			v_cur_2 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic2, "cur")
		else:
			v_cur_1 = self.critic1(batch.c_in_cur, None)[0]
			v_cur_2 = self.critic2(batch.c_in_cur, None)[0]
		critic_loss = F.mse_loss(v_cur_1.reshape(*pre_sz), target_q, reduce=False) + \
			F.mse_loss(v_cur_2.reshape(*pre_sz), target_q, reduce=False)

		# add sirnn extra loss
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn" \
			and self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
			v_cur_sirnn_1 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic_sirnn_1, "cur")
			v_cur_sirnn_2 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic_sirnn_2, "cur")
			sirnn_loss = F.mse_loss(v_cur_sirnn_1.reshape(*pre_sz), target_q, reduce=False) + \
				F.mse_loss(v_cur_sirnn_2.reshape(*pre_sz), target_q, reduce=False)
			critic_loss += sirnn_loss
			self.record("learn/loss_critic_before_sirnn", critic_loss.detach().mean().item())
			self.record("learn/loss_critic_sirnn", sirnn_loss.detach().mean().item())

		# use mask
		critic_loss = apply_mask(critic_loss, batch.valid_mask).mean()
		self.record("learn/loss_critic", critic_loss.item())
		self.record("learn/loss_critic_normed", critic_loss.item()/batch.valid_mask.float().mean().item())
		self.record("learn/valid_mask_ratio", batch.valid_mask.float().mean())

		self.critic1_optim.zero_grad()
		self.critic2_optim.zero_grad()
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn" \
			and self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
			self.critic_sirnn_1_optim.zero_grad()
			self.critic_sirnn_2_optim.zero_grad()
		critic_loss.backward()
		self.critic1_optim.step()
		self.critic2_optim.step()
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn" \
			and self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
			self.critic_sirnn_1_optim.step()
			self.critic_sirnn_2_optim.step()

		return {
			"critic_loss": critic_loss.cpu().item()
		}

	def update_actor(self, batch):
		res_info = {}
		combined_loss = 0.
		
		### actor loss
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			if self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
				current_q1a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic_sirnn_1, "cur")
				current_q2a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic_sirnn_2, "cur")
			else:
				current_q1a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic1, "cur")
				current_q2a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic2, "cur")
		else:
			current_q1a, _ = self.critic1(batch.c_in_online_cur, None)
			current_q2a, _ = self.critic2(batch.c_in_online_cur, None)
		actor_loss = self._log_alpha.exp().detach() * batch.logprob_online_cur - torch.min(current_q1a, current_q2a)
		actor_loss = apply_mask(actor_loss, batch.valid_mask).mean()
		combined_loss += actor_loss
		self.record("learn/loss_actor", actor_loss.item())
		self.record("learn/loss_actor_normed", actor_loss.item()/batch.valid_mask.float().mean().item())

		# add obs_pred loss
		if self.global_cfg.actor_input.obs_pred.turn_on:
			combined_loss = self.add_obs_pred_loss(batch, combined_loss)
		
		# add obs_encode loss
		if self.global_cfg.actor_input.obs_encode.turn_on:
			combined_loss = self.add_obs_encode_loss(batch, combined_loss)

		# backward and optim
		self.actor_optim.zero_grad()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.zero_grad()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.zero_grad()
		combined_loss.backward()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.step()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.step()
		self.actor_optim.step()

		# update alpha (use batch.logprob_online_cur)

		if self._is_auto_alpha:
			if self.global_cfg.debug.use_log_alpha_for_mul_logprob:
				alpha_mul = self._log_alpha
			else:
				alpha_mul = self._log_alpha.exp()
			
			if self.global_cfg.debug.entropy_mask_loss_renorm:
				cur_entropy = - apply_mask(batch.logprob_online_cur.detach(), batch.valid_mask).mean() / batch.valid_mask.mean() # (*, 1)
				alpha_loss = - alpha_mul * (self._target_entropy - cur_entropy)
			else:
				cur_entropy = - batch.logprob_online_cur.detach() # (*, 1)
				alpha_loss = - alpha_mul * apply_mask(self._target_entropy-cur_entropy, batch.valid_mask) # (*, 1)
				alpha_loss = alpha_loss.mean()
			
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self.record("learn/alpha_loss_normed", alpha_loss.item() / batch.valid_mask.float().mean().item())
			self.record("learn/alpha", self._log_alpha.exp().detach().cpu().item())
			self.record("learn/log_alpha", self._log_alpha.detach().cpu().item())
			self.record("learn/entropy", cur_entropy.mean().cpu().item())
			self.record("learn/entropy_target", self._target_entropy)
		
		return {
			"actor_loss": actor_loss.cpu().item()
		}

	def get_act_online(self, batch):
		# cur
		if self.global_cfg.actor_input.history_merge_method == "stack_rnn" and self._burnin_num():
			if self.cfg.global_cfg.debug.rnn_turn_on_burnin:
				mu, var = forward_with_burnin(
					input=batch.a_in_cur,
					burnin_input=batch.burnin_a_in,
					net=self.actor,
					remaster_mask=batch.burnin_remaster_mask,
					remaster_mode="cur"
				)
			else:
				(mu, var), _ = self.actor(batch.a_in_cur, None)
		else:
			(mu, var), _ = self.actor(batch.a_in_cur, None)
		act_online_cur, logprob_online_cur = self.actor.sample_act(mu, var)
		
		# next
		with torch.no_grad():
			if self.global_cfg.actor_input.history_merge_method == "stack_rnn" and self._burnin_num():
				if self.cfg.global_cfg.debug.rnn_turn_on_burnin:
					mu, var = forward_with_burnin(
						input=batch.a_in_next,
						burnin_input=batch.burnin_a_in,
						net=self.actor,
						remaster_mask=batch.burnin_remaster_mask,
						remaster_mode="next"
					)
				else:
					(mu, var), _ = self.actor(batch.a_in_next, state=None)
			else:
				(mu, var), _ = self.actor(batch.a_in_next, state=None)
			act_online_next, logprob_online_next = self.actor.sample_act(mu, var)
		return act_online_cur, act_online_next, {
			"logprob_online_cur": logprob_online_cur,
			"logprob_online_next": logprob_online_next
		}
	
	def _mse_optimizer(self,
			batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
		) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		current_q, critic_state = critic(batch.critic_input_cur_offline)
		target_q = torch.tensor(batch.returns).to(current_q.device)
		td = current_q.flatten() - target_q.flatten()
		critic_loss = (
			(td.pow(2) * weight) * batch.valid_mask.flatten()
		).mean()
		critic_loss = (td.pow(2) * weight)
		critic_loss = critic_loss.mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss

	def _init_sac_alpha(self):
		"""
		init self._log_alpha, self._alpha_optim, self._is_auto_alpha, self._target_entropy
		"""
		cfg = self.cfg
		if isinstance(cfg.policy.alpha, Iterable):
			self._is_auto_alpha = True
			self._target_entropy, self._log_alpha, self._alpha_optim = cfg.policy.alpha
			if type(self._target_entropy) == str and self._target_entropy == "neg_act_num":
				self._target_entropy = - np.prod(self.env.action_space.shape)
			elif type(self._target_entropy) == float:
				self._target_entropy = torch.tensor(self._target_entropy).to(self.device)
			else: 
				raise ValueError("Invalid target entropy type.")
			assert cfg.policy.alpha[1].shape == torch.Size([1]) and cfg.policy.alpha[1].requires_grad
			self._alpha_optim = self._alpha_optim([self._log_alpha])
		elif isinstance(cfg.policy.alpha, float):
			self._is_auto_alpha = False
			self._log_alpha = cfg.policy.alpha # here, the cfg alpha is actually log_alpha
		else: 
			raise ValueError("Invalid alpha type.")

class DDPGRunner(TD3SACRunner):
	ALGORITHM = "ddpg"
	
	def update_critic(self, batch):
		pre_sz = list(batch.done.shape)
		
		if self.cfg.global_cfg.debug.use_terminated_mask_for_value:
			value_mask = batch.terminated
		else:
			value_mask = batch.done
		
		with torch.no_grad():
			target_q = (batch.rew + self.cfg.policy.gamma * (1 - value_mask.int()) * \
				self.critic1_old(batch.c_in_online_next, None)[0].squeeze(-1)
			).reshape(*pre_sz,-1)
		
		critic_loss = F.mse_loss(
			self.critic1(batch.c_in_cur, None)[0].reshape(*pre_sz,-1), 
			target_q, 
			reduce=False
		)
			
		critic_loss = apply_mask(critic_loss, batch.valid_mask).mean()
		self.record("learn/critic_loss", critic_loss.cpu().item())
		self.record("learn/valid_mask_ratio", batch.valid_mask.float().mean())
		
		self.critic1_optim.zero_grad()
		critic_loss.backward()
		self.critic1_optim.step()

		return {
			"critic_loss": critic_loss.cpu().item()
		}

	def update_actor(self, batch):
		actor_loss, _ = self.critic1(batch.c_in_online_cur, None)
		actor_loss =  - apply_mask(actor_loss, batch.valid_mask).mean()
		combined_loss = 0. + actor_loss

		# add obs_pred loss
		if self.global_cfg.actor_input.obs_pred.turn_on:
			combined_loss = self.add_obs_pred_loss(batch, combined_loss)
		
		# add obs_encode loss
		if self.global_cfg.actor_input.obs_encode.turn_on:
			combined_loss = self.add_obs_encode_loss(batch, combined_loss)

		# backward and optim
		self.actor_optim.zero_grad()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.zero_grad()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.zero_grad()
		combined_loss.backward()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.step()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.step()
		self.actor_optim.step()
		self.record("learn/actor_loss", actor_loss.cpu().item())

	def get_act_online(self, batch):
		act_online_cur = self.actor(
			batch.a_in_cur, 
			state=batch.a_state_cur if self._burnin_num() else None
		)[0][0]
		with torch.no_grad():
			act_online_next = self.actor_old(
				batch.a_in_next, 
				state=batch.a_state_next if self._burnin_num() else None
			)[0][0]
		return act_online_cur, act_online_next, {}

