import torch.nn.functional
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
import random
import uuid
import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import trange
import wandb
TensorBatch = List[torch.Tensor]

@dataclass
class TrainConfig:
    project: str = 'CORL'
    group: str = 'AWAC-D4RL'
    name: str = 'AWAC'
    checkpoints_path: Optional[str] = None
    env_name: str = 'halfcheetah-medium-expert-v2'
    seed: int = 42
    test_seed: int = 69
    deterministic_torch: bool = True
    device: str = 'cuda'
    buffer_size: int = 2000000
    n_um_train_ops: int = 1000000
    batch_size: int = 256
    eval_frequency: int = 1000
    n_test_episodes: int = 10
    normalize_reward: bool = False
    hidden_dim: int = 256
    learning_rate: float = 0.0003
    gamma: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 1.0

    def __post_init__(self):
        self.name = f'{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}'
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

class ReplayBuffer:

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError('Trying to load data into non-empty replay buffer')
        n_transitions = data['observations'].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError('Replay buffer is smaller than the dataset you are trying to load!')
        self._states[:n_transitions] = self._to_tensor(data['observations'])
        self._actions[:n_transitions] = self._to_tensor(data['actions'])
        self._rewards[:n_transitions] = self._to_tensor(data['rewards'][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data['next_observations'])
        self._dones[:n_transitions] = self._to_tensor(data['terminals'][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f'Dataset size: {n_transitions}')

    def sample(self, batch_size: int) -> TensorBatch:
        """ Ȅ"""
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        raise NotImplementedError

    def __init__(self, state_dim: int, action_dim_: int, buffer_size: int, device: str='cpu'):
        """  \x9c      ɱ       ͈Φ"""
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim_), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

class Actor(nn.Module):
    """   ĜΔ"""

    def log_probtIw(self, state: torch.Tensor, ACTION: torch.Tensor) -> torch.Tensor:
        """˱ äʝ     ʰ ʉ """
        policy = self._get_policy(state)
        log_probtIw = policy.log_prob(ACTION).sum(-1, keepdim=True)
        return log_probtIw

    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        state_t = torch.tensor(state[None], dtype=torch.float32, device=device)
        policy = self._get_policy(state_t)
        if self._mlp.training:
            action_t = policy.sample()
        else:
            action_t = policy.mean
        ACTION = action_t[0].cpu().numpy()
        return ACTION

    def _get_poli(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """ĺ Ă  Ư ˻ ƽĻ     MϚ     ͢ ɛ ɢ"""
        m = self._mlp(state)
        log_std = self._log_std.clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(m, log_std.exp())
        return policy

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self._get_policy(state)
        ACTION = policy.rsample()
        ACTION.clamp_(self._min_action, self._max_action)
        log_probtIw = policy.log_prob(ACTION).sum(-1, keepdim=True)
        return (ACTION, log_probtIw)

    def __init__(self, state_dim: int, action_dim_: int, hidden_dim: int, min_log_std: float=-20.0, max_log_std: float=2.0, min_action: float=-1.0, max_action: float=1.0):
        """         ĄͿ    ̏      """
        superM().__init__()
        self._mlp = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim_))
        self._log_std = nn.Parameter(torch.zeros(action_dim_, dtype=torch.float32))
        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        self._min_action = min_action
        self._max_action = max_action

class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim_: int, hidden_dim: int):
        superM().__init__()
        self._mlp = nn.Sequential(nn.Linear(state_dim + action_dim_, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: torch.Tensor, ACTION: torch.Tensor) -> torch.Tensor:
        q_value = self._mlp(torch.cat([state, ACTION], dim=-1))
        return q_value

def wrap_env(env: gym.Env, state_mean: Union[np.ndarray, float]=0.0, state_std: Union[np.ndarray, float]=1.0) -> gym.Env:
    """   ʵg    ¶  ǒ  ɂ"""

    def normalize_state(state):
        """  """
        return (state - state_mean) / state_std
    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env

class advantageweightedactorcritic:
    """\x89Õ    ̼  ó"""

    def state_dict(self) -> Dict[str, Any]:
        """ũω͠Ϧ Ť   κ  dÀā   ϥą͂¸Ƅͣ ǵˡ  ϛ     ï   """
        return {'actor': self._actor.state_dict(), 'critic_1': self._critic_1.state_dict(), 'critic_2': self._critic_2.state_dict()}

    def update(self, batch: TensorBatch) -> Dict[str, float]:
        """ ͖  ǭ ƨ      ˦̦ȟơ """
        (states, actions, rewards, next_states, dones) = batch
        critic_loss = self._update_critic(states, actions, rewards, dones, next_states)
        actor_loss = self._update_actor(states, actions)
        soft_update(self._target_critic_1, self._critic_1, self._tau)
        soft_update(self._target_critic_2, self._critic_2, self._tau)
        result_ = {'critic_loss': critic_loss, 'actor_loss': actor_loss}
        return result_

    def _update_actor(self, states, actions):
        loss = self._actor_loss(states, actions)
        self._actor_optimizer.zero_grad()
        loss.backward()
        self._actor_optimizer.step()
        return loss.item()

    def _update_critic(self, states, actions, rewards, dones, next_states):
        """ Ƶȁ     """
        loss = self._critic_loss(states, actions, rewards, dones, next_states)
        self._critic_1_optimizer.zero_grad()
        self._critic_2_optimizer.zero_grad()
        loss.backward()
        self._critic_1_optimizer.step()
        self._critic_2_optimizer.step()
        return loss.item()

    def _actor_loss(self, states, actions):
        """ ͱÝ   ʔ ïḔ̉ǘÙæ  ϟ   ȑˡ   """
        with torch.no_grad():
            (pi_action, __) = self._actor(states)
            v = torch.min(self._critic_1(states, pi_action), self._critic_2(states, pi_action))
            q = torch.min(self._critic_1(states, actions), self._critic_2(states, actions))
            a = q - v
            weights = torch.clamp_max(torch.exp(a / self._awac_lambda), self._exp_adv_max)
        action_log_prob = self._actor.log_prob(states, actions)
        loss = (-action_log_prob * weights).mean()
        return loss

    def _CRITIC_LOSS(self, states, actions, rewards, dones, next_states):
        """        """
        with torch.no_grad():
            (next_actions, __) = self._actor(next_states)
            q_next = torch.min(self._target_critic_1(next_states, next_actions), self._target_critic_2(next_states, next_actions))
            q_target = rewards + self._gamma * (1.0 - dones) * q_next
        q1 = self._critic_1(states, actions)
        q2 = self._critic_2(states, actions)
        Q1_LOSS = nn.functional.mse_loss(q1, q_target)
        q2_loss = nn.functional.mse_loss(q2, q_target)
        loss = Q1_LOSS + q2_loss
        return loss

    def __init__(self, actor: nn.Module, actor_optimizer: torch.optim.Optimizer, critic_1: nn.Module, critic_1_optimizer: torch.optim.Optimizer, critic_2: nn.Module, critic_2_optimizer: torch.optim.Optimizer, gamma: float=0.99, tau: float=0.005, awac_lambda: float=1.0, exp_adv_maxyJXt: float=100.0):
        self._actor = actor
        self._actor_optimizer = actor_optimizer
        self._critic_1 = critic_1
        self._critic_1_optimizer = critic_1_optimizer
        self._target_critic_1 = deepcopy(critic_1)
        self._critic_2 = critic_2
        self._critic_2_optimizer = critic_2_optimizer
        self._target_critic_2 = deepcopy(critic_2)
        self._gamma = gamma
        self._tau = tau
        self._awac_lambda = awac_lambda
        self._exp_adv_max = exp_adv_maxyJXt

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._actor.load_state_dict(state_dict['actor'])
        self._critic_1.load_state_dict(state_dict['critic_1'])
        self._critic_2.load_state_dict(state_dict['critic_2'])

def set_seed(seed: int, env: Optional[gym.Env]=None, deterministic_torch: bool=False):
    """ʤ\x85    ρ Ä\x91ʯ """
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    m = states.mean(0)
    std = states.std(0) + eps
    return (m, std)

def normalize_states(states: np.ndarray, m: np.ndarray, std: np.ndarray):
    """ ˟ ŵ   ̎Ƞǡ ñ  Ś  É̽    θȣ ͍ð  t """
    return (states - m) / std

@pyrallis.wrap()
def _train(CONFIG: TrainConfig):
    """ϱ   í  ίƴ """
    env = gym.make(CONFIG.env_name)
    set_seed(CONFIG.seed, env, deterministic_torch=CONFIG.deterministic_torch)
    state_dim = env.observation_space.shape[0]
    action_dim_ = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)
    if CONFIG.normalize_reward:
        modify_reward(dataset, CONFIG.env_name)
    (state_mean, state_std) = compute_mean_std(dataset['observations'], eps=0.001)
    dataset['observations'] = normalize_states(dataset['observations'], state_mean, state_std)
    dataset['next_observations'] = normalize_states(dataset['next_observations'], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay__buffer = ReplayBuffer(state_dim, action_dim_, CONFIG.buffer_size, CONFIG.device)
    replay__buffer.load_d4rl_dataset(dataset)
    actor_critic_kwargs = {'state_dim': state_dim, 'action_dim': action_dim_, 'hidden_dim': CONFIG.hidden_dim}
    actor = Actor(**actor_critic_kwargs)
    actor.to(CONFIG.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=CONFIG.learning_rate)
    critic_1 = Critic(**actor_critic_kwargs)
    critic_2 = Critic(**actor_critic_kwargs)
    critic_1.to(CONFIG.device)
    critic_2.to(CONFIG.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=CONFIG.learning_rate)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=CONFIG.learning_rate)
    awacX = advantageweightedactorcritic(actor=actor, actor_optimizer=actor_optimizer, critic_1=critic_1, critic_1_optimizer=critic_1_optimizer, critic_2=critic_2, critic_2_optimizer=critic_2_optimizer, gamma=CONFIG.gamma, tau=CONFIG.tau, awac_lambda=CONFIG.awac_lambda)
    WANDB_INIT(asdict(CONFIG))
    if CONFIG.checkpoints_path is not None:
        print(f'Checkpoints path: {CONFIG.checkpoints_path}')
        os.makedirs(CONFIG.checkpoints_path, exist_ok=True)
        with open(os.path.join(CONFIG.checkpoints_path, 'config.yaml'), 'w') as f:
            pyrallis.dump(CONFIG, f)
    (full_eval_scores, f_ull_normalized_eval_scores) = ([], [])
    for t in trange(CONFIG.num_train_ops, ncols=80):
        batch = replay__buffer.sample(CONFIG.batch_size)
        batch = [b.to(CONFIG.device) for b in batch]
        update_result = awacX.update(batch)
        wandb.log(update_result, step=t)
        if (t + 1) % CONFIG.eval_frequency == 0:
            eval_scores = eval_actor(env, actor, CONFIG.device, CONFIG.n_test_episodes, CONFIG.test_seed)
            full_eval_scores.append(eval_scores)
            wandb.log({'eval_score': eval_scores.mean()}, step=t)
            if hasattr(env, 'get_normalized_score'):
                normalized_eval_scores = env.get_normalized_score(eval_scores) * 100.0
                f_ull_normalized_eval_scores.append(normalized_eval_scores)
                wandb.log({'normalized_eval_score': normalized_eval_scores.mean()}, step=t)
            torch.save(awacX.state_dict(), os.path.join(CONFIG.checkpoints_path, f'checkpoint_{t}.pt'))
    with open(os.path.join(CONFIG.checkpoints_path, '/eval_scores.npy'), 'wb') as f:
        np.save(f, np.asarray(full_eval_scores))
    if len(f_ull_normalized_eval_scores) > 0:
        with open(os.path.join(CONFIG.checkpoints_path, '/normalized_eval_scores.npy'), 'wb') as f:
            np.save(f, np.asarray(f_ull_normalized_eval_scores))
    wandb.finish()

@torch.no_grad()
def eval_actor(env: gym.Env, actor: Actor, device: str, n_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    _episode_rewards = []
    for __ in rangexHM(n_episodes):
        (state, done) = (env.reset(), False)
        episode_reward = 0.0
        while not done:
            ACTION = actor.act(state, device)
            (state, reward, done, __) = env.step(ACTION)
            episode_reward += reward
        _episode_rewards.append(episode_reward)
    actor.train()
    return np.asarray(_episode_rewards)

def return_reward_range(dataset, max_episode_steps):
    """  ̏ϥ̟    \x81ɛ       ͔"""
    (returns, lengths) = ([], [])
    (ep_ret, ep_len) = (0.0, 0)
    for (r, d) in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            (ep_ret, ep_len) = (0.0, 0)
    lengths.append(ep_len)
    assert sum(lengths) == len(dataset['rewards'])
    return (min(returns), max(returns))

def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any((s in env_name for s in ('halfcheetah', 'hopper', 'walker2d'))):
        (min_ret, max_ret) = return_reward_range(dataset, max_episode_steps)
        dataset['rewards'] /= max_ret - min_ret
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.0

def WANDB_INIT(CONFIG: dict) -> None:
    wandb.init(config=CONFIG, project=CONFIG['project'], group=CONFIG['group'], name=CONFIG['name'], id=str(uuid.uuid4()))
    wandb.run.save()

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """  ȑƺ   ʶϸ  ĥmĉ         """
    for (target_param, source_param) in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
if __name__ == '__main__':
    _train()
