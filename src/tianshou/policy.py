import warnings
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.exploration import BaseNoise, GaussianNoise
from tianshou.policy import BasePolicy

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.utils import MultipleLRSchedulers

class DDPGPolicy(BasePolicy):
	"""Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971.

	:param torch.nn.Module actor: the actor network following the rules in
		:class:`~tianshou.policy.BasePolicy`. (s -> logits)
	:param torch.optim.Optimizer actor_optim: the optimizer for actor network.
	:param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
	:param torch.optim.Optimizer critic_optim: the optimizer for critic network.
	:param float tau: param for soft update of the target network. Default to 0.005.
	:param float gamma: discount factor, in [0, 1]. Default to 0.99.
	:param BaseNoise exploration_noise: the exploration noise,
		add to the action. Default to ``GaussianNoise(sigma=0.1)``.
	:param bool reward_normalization: normalize the reward to Normal(0, 1),
		Default to False.
	:param int estimation_step: the number of steps to look ahead. Default to 1.
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
		actor: Optional[torch.nn.Module],
		actor_optim: Optional[torch.optim.Optimizer],
		critic: Optional[torch.nn.Module],
		critic_optim: Optional[torch.optim.Optimizer],
		tau: float = 0.005,
		gamma: float = 0.99,
		exploration_noise: Optional[BaseNoise] = GaussianNoise(sigma=0.1),
		reward_normalization: bool = False,
		estimation_step: int = 1,
		action_scaling: bool = True,
		action_bound_method: str = "clip",
		cfg=None,
		**kwargs: Any,
	) -> None:
		super().__init__(
			action_scaling=action_scaling,
			action_bound_method=action_bound_method,
			**kwargs
		)
		self.cfg = cfg
		assert action_bound_method != "tanh", "tanh mapping is not supported" \
			"in policies where action is used as input of critic , because" \
			"raw action in range (-inf, inf) will cause instability in training"
		if actor is not None and actor_optim is not None:
			self.actor: torch.nn.Module = actor
			self.actor_old = deepcopy(actor)
			self.actor_old.eval()
			self.actor_optim: torch.optim.Optimizer = actor_optim
		if critic is not None and critic_optim is not None:
			self.critic: torch.nn.Module = critic
			self.critic_old = deepcopy(critic)
			self.critic_old.eval()
			self.critic_optim: torch.optim.Optimizer = critic_optim
		assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
		self.tau = tau
		assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
		self._gamma = gamma
		self._noise = exploration_noise
		# it is only a little difference to use GaussianNoise
		# self.noise = OUNoise()
		self._rew_norm = reward_normalization
		self._n_step = estimation_step

	def set_exp_noise(self, noise: Optional[BaseNoise]) -> None:
		"""Set the exploration noise."""
		self._noise = noise

	def train(self, mode: bool = True) -> "DDPGPolicy":
		"""Set the module in training mode, except for the target network."""
		self.training = mode
		self.actor.train(mode)
		self.critic.train(mode)
		return self

	def sync_weight(self) -> None:
		"""Soft-update the weight for the target network."""
		self.soft_update(self.actor_old, self.actor, self.tau)
		self.soft_update(self.critic_old, self.critic, self.tau)

	def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
		batch = buffer[indices]  # batch.obs_next: s_{t+n}
		target_q = self.critic_old(
			batch.obs_next,
			self(batch, model='actor_old', input='obs_next').act
		)
		return target_q

	def process_fn(
		self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
	) -> Batch:
		if self.cfg.debug.return_type == "tianshou":
			batch = self.compute_nstep_return(
				batch, buffer, indices, self._target_q, self._gamma, self._n_step,
				self._rew_norm
			)
		elif self.cfg.debug.return_type == "tianshou_onestep":
			batch = self.compute_onestep_return(
				batch, buffer, indices, self._target_q, self._gamma, self._n_step,
				self._rew_norm
			)
		elif self.cfg.debug.return_type == "my_use_done":
			with torch.no_grad():
				act_next = self(batch, model='actor_old', input='obs_next').act
				rew = torch.tensor(batch.rew).to(act_next.device).unsqueeze(-1)
				valid_mask = torch.tensor(batch.done).to(act_next.device).int().unsqueeze(-1)
				obs_next = torch.tensor(batch.obs_next).to(act_next.device)
				batch.returns = rew + (1 - valid_mask) * self.critic_old(
					obs_next, act_next
				) * self._gamma
		elif self.cfg.debug.return_type == "my_use_terminated":
			with torch.no_grad():
				act_next = self(batch, model='actor_old', input='obs_next').act
				rew = torch.tensor(batch.rew).to(act_next.device).unsqueeze(-1)
				valid_mask = torch.tensor(batch.terminated).to(act_next.device).int().unsqueeze(-1)
				obs_next = torch.tensor(batch.obs_next).to(act_next.device)
				batch.returns = rew + (1 - valid_mask) * self.critic_old(
					obs_next, act_next
				) * self._gamma

		elif self.cfg.debug.return_type == "eval_all":
			with torch.no_grad():
				batch_tmp = self.compute_nstep_return(
					batch, buffer, indices, self._target_q, self._gamma, self._n_step,
					self._rew_norm
				)
				returns_tianshou = batch_tmp.returns
				del batch_tmp
			with torch.no_grad():
				batch_tmp = self.compute_onestep_return(
					batch, buffer, indices, self._target_q, self._gamma, self._n_step,
					self._rew_norm
				)
				returns_tianshou_onestep = batch_tmp.returns
			with torch.no_grad():
				act_next = self(batch, model='actor_old', input='obs_next').act
				rew = torch.tensor(batch.rew).to(act_next.device).unsqueeze(-1)
				valid_mask = torch.tensor(batch.done).to(act_next.device).int().unsqueeze(-1)
				obs_next = torch.tensor(batch.obs_next).to(act_next.device)
				returns_my_use_terminated = rew + (1 - valid_mask) * self.critic_old(
					obs_next, act_next
				) * self._gamma
			with torch.no_grad():
				act_next = self(batch, model='actor_old', input='obs_next').act
				rew = torch.tensor(batch.rew).to(act_next.device).unsqueeze(-1)
				valid_mask = torch.tensor(batch.done).to(act_next.device).int().unsqueeze(-1)
				obs_next = torch.tensor(batch.obs_next).to(act_next.device)
				returns_my_use_done = rew + (1 - valid_mask) * self.critic_old(
					obs_next, act_next
				) * self._gamma


			assert (returns_tianshou == returns_tianshou_onestep).all(), "returns_tianshou != returns_tianshou_onestep"
			# assert torch.allclose(returns_tianshou, returns_my_use_done.float()), "returns_tianshou != returns_my_use_done"
			if len(torch.where((returns_tianshou-returns_my_use_done.float())> 0.00001)[0].shape) > 0:
				print(torch.where((returns_tianshou-returns_my_use_done.float())> 0.00001))
			# print("returns_tianshou != returns_my_use_done")
			# diff_indices = torch.where(returns_my_use_done != returns_tianshou)
			# print(diff_indices)
			# print("returns_tianshou != returns_my_use_terminated")
			# diff_indices = torch.where(returns_my_use_terminated != returns_tianshou)
			# print(diff_indices)
			# print("returns_my_use_done != returns_my_use_terminated")
			# diff_indices = torch.where(returns_my_use_done != returns_my_use_terminated)

			# should be the same (value mask is terminated)
			assert (~batch.terminated == BasePolicy.value_mask(buffer, indices)).all(), "terminated should be the same"
			
			batch.returns = returns_tianshou
		else:
			raise ValueError("Invalid return type {}".format(self.cfg.debug.return_type))
		
		return batch

	@staticmethod
	def compute_nstep_return(
		batch: Batch,
		buffer: ReplayBuffer,
		indice: np.ndarray,
		target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
		gamma: float = 0.99,
		n_step: int = 1,
		rew_norm: bool = False,
	) -> Batch:
		r"""Compute n-step return for Q-learning targets.

		.. math::
			G_t = \sum_{i = t}^{t + n - 1} \gamma^{i - t}(1 - d_i)r_i +
			\gamma^n (1 - d_{t + n}) Q_{\mathrm{target}}(s_{t + n})

		where :math:`\gamma` is the discount factor, :math:`\gamma \in [0, 1]`,
		:math:`d_t` is the done flag of step :math:`t`.

		:param Batch batch: a data batch, which is equal to buffer[indice].
		:param ReplayBuffer buffer: the data buffer.
		:param function target_q_fn: a function which compute target Q value
			of "obs_next" given data buffer and wanted indices.
		:param float gamma: the discount factor, should be in [0, 1]. Default to 0.99.
		:param int n_step: the number of estimation step, should be an int greater
			than 0. Default to 1.
		:param bool rew_norm: normalize the reward to Normal(0, 1), Default to False.

		:return: a Batch. The result will be stored in batch.returns as a
			torch.Tensor with the same shape as target_q_fn's return tensor.
		"""
		assert not rew_norm, \
			"Reward normalization in computing n-step returns is unsupported now."
		rew = buffer.rew
		bsz = len(indice)
		indices = [indice]
		for _ in range(n_step - 1):
			indices.append(buffer.next(indices[-1]))
		indices = np.stack(indices)
		# terminal indicates buffer indexes nstep after 'indice',
		# and are truncated at the end of each episode
		terminal = indices[-1]
		with torch.no_grad():
			target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
		target_q = to_numpy(target_q_torch.reshape(bsz, -1))
		target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
		end_flag = buffer.done.copy()
		end_flag[buffer.unfinished_index()] = True
		target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)

		batch.returns = to_torch_as(target_q, target_q_torch)
		if hasattr(batch, "weight"):  # prio buffer update
			batch.weight = to_torch_as(batch.weight, target_q_torch)
		return batch

	@staticmethod
	def compute_onestep_return(
		batch: Batch,
		buffer: ReplayBuffer,
		indice: np.ndarray,
		target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
		gamma: float = 0.99,
		n_step: int = 1,
		rew_norm: bool = False,
	) -> Batch:
		assert not rew_norm, \
			"Reward normalization in computing n-step returns is unsupported now."
		rew = buffer.rew
		bsz = len(indice)
		indices = [indice]
		indices = np.stack(indices)
		# terminal indicates buffer indexes nstep after 'indice',
		# and are truncated at the end of each episode
		terminal = indices[-1] # == indice when n_step == 1 # (256,)
		with torch.no_grad():
			target_q_torch = target_q_fn(buffer, terminal)  # (bsz, ?)
		target_q = to_numpy(target_q_torch.reshape(bsz, -1))
		target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
		end_flag = buffer.done.copy()
		end_flag[buffer.unfinished_index()] = True
		target_q = _onestep_return(rew, end_flag, target_q, indices, gamma, n_step) # (1000000,)

		batch.returns = to_torch_as(target_q, target_q_torch)
		if hasattr(batch, "weight"):  # prio buffer update
			batch.weight = to_torch_as(batch.weight, target_q_torch)
		return batch

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

	@staticmethod
	def _mse_optimizer(
		batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		current_q = critic(batch.obs, batch.act).flatten()
		target_q = batch.returns.flatten()
		td = current_q - target_q
		# critic_loss = F.mse_loss(current_q1, target_q)
		critic_loss = (td.pow(2) * weight).mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss

	def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
		# DEBUG = 1
		# if DEBUG:
		# 	act_next = self(batch, model='actor_old', input='obs_next').act
		# 	batch.to_torch(dtype=torch.float32, device=act_next.device)
		# 	target_q_2 = batch.rew + (1-batch.done) * self.critic_old(batch.obs_next, act_next.flatten(end_dim=-2)).flatten() * 0.99
		# 	np.arange(256)[(batch.returns.flatten() != target_q_2).cpu()]
		# 	np.arange(256)[batch.done.cpu() == True]
		# critic
		td, critic_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
		batch.weight = td  # prio-buffer
		# actor
		actor_loss = -self.critic(batch.obs, self(batch).act).mean()
		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()
		self.sync_weight()
		return {
			"loss/actor": actor_loss.item(),
			"loss/critic": critic_loss.item(),
		}

	def exploration_noise(self, act: Union[np.ndarray, Batch],
						  batch: Batch) -> Union[np.ndarray, Batch]:
		if self._noise is None:
			return act
		if isinstance(act, np.ndarray):
			return act + self._noise(act.shape)
		warnings.warn("Cannot add exploration noise to non-numpy_array action.")
		return act


def _nstep_return(
	rew: np.ndarray,
	end_flag: np.ndarray,
	target_q: np.ndarray,
	indices: np.ndarray,
	gamma: float,
	n_step: int,
) -> np.ndarray:
	gamma_buffer = np.ones(n_step + 1)
	for i in range(1, n_step + 1):
		gamma_buffer[i] = gamma_buffer[i - 1] * gamma
	target_shape = target_q.shape
	bsz = target_shape[0]
	# change target_q to 2d array
	target_q = target_q.reshape(bsz, -1)
	returns = np.zeros(target_q.shape) # B, T
	gammas = np.full(indices[0].shape, n_step) # B
	for n in range(n_step - 1, -1, -1):
		now = indices[n]
		gammas[end_flag[now] > 0] = n + 1
		returns[end_flag[now] > 0] = 0.0
		returns = rew[now].reshape(bsz, 1) + gamma * returns
	target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
	return target_q.reshape(target_shape)

def _onestep_return(
	rew: np.ndarray,
	end_flag: np.ndarray,
	target_q: np.ndarray,
	indices: np.ndarray,
	gamma: float,
	n_step: int,
) -> np.ndarray:
	gamma_buffer = np.ones(n_step + 1)
	gamma_buffer[1] = gamma_buffer[0] * gamma
	target_shape = target_q.shape
	bsz = target_shape[0]
	# change target_q to 2d array
	target_q = target_q.reshape(bsz, -1)
	gammas = np.full(indices[0].shape, n_step) # B
	now = indices[0]
	gammas[end_flag[now] > 0] = 1
	returns = rew[now].reshape(bsz, 1)
	target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
	return target_q.reshape(target_shape)
