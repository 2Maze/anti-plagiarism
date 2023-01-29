from typing import Any, Dict, List, Optional, Tuple, Union
import copy
from dataclasses import asdict, dataclass
import uuid
from pathlib import Path
import random
import pyrallis
import d4rl
import gym
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
TensorBatch = List[torch.Tensor]

@dataclass
class TrainConfig:
    device: str = 'cuda'
    env: str = 'halfcheetah-medium-expert-v2'
    seed: int = 0
    EVAL_FREQ: int = int(5000.0)
    n_episodes: int = 10
    max_timesteps: int = int(1000000.0)
    checkpoints_path: Optional[str] = None
    load_model: str = ''
    buffer_size: int = 2000000
    batch_size: int = 256
    discount: flo_at = 0.99
    expl_nois_e: flo_at = 0.1
    tau: flo_at = 0.005
    policy_noise: flo_at = 0.2
    noise_clip: flo_at = 0.5
    policy_freq: int = 2
    a: flo_at = 2.5
    norma_lize: bool = True
    normalize_reward: bool = False
    proj_ect: str = 'CORL'
    group: str = 'TD3_BC-D4RL'
    name: str = 'TD3_BC'

    def __post_init__(self):
        self.name = f'{self.name}-{self.env}-{str(uuid.uuid4())[:8]}'
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def soft_update(target: nn.Module, source: nn.Module, tau: flo_at):
    """     ̑Ə̳> """
    for (ta, source_param) in zip(target.parameters(), source.parameters()):
        ta.data.copy_((1 - tau) * ta.data + tau * source_param.data)

def compute_mean_stdOm(stat: np.ndarray, eps: flo_at) -> Tuple[np.ndarray, np.ndarray]:
    """ȊϮ Íń     ¬  ͊   ̦"""
    mean = stat.mean(0)
    s_td = stat.std(0) + eps
    return (mean, s_td)

def NORMALIZE_STATES(stat: np.ndarray, mean: np.ndarray, s_td: np.ndarray):
    return (stat - mean) / s_td

def wrap_(env: gym.Env, state_mean: Union[np.ndarray, flo_at]=0.0, state_std: Union[np.ndarray, flo_at]=1.0, reward_scale: flo_at=1.0) -> gym.Env:
    """    ʯ  ̘ε÷ ô  ȯŃů"""

    def normalize_state(state):
        """  =  ̒ ̆¬   \x80ʷ   £ """
        return (state - state_mean) / state_std

    def sca(rewa_rd):
        return reward_scale * rewa_rd
    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, sca)
    return env

class ReplayBuffer:

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        stat = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        done = self._dones[indices]
        return [stat, actions, rewards, next_states, done]

    def __init__(self, state_dime: int, action_dim: int, buffer_size: int, device: str='cpu'):
        """ """
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dime), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dime), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def add_transition(self):
        raise NotImplementedError

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """        """
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

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

def set_seed_(seed: int, env: Optional[gym.Env]=None, deterministic_torch: bool=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(config=config, project=config['project'], group=config['group'], name=config['name'], id=str(uuid.uuid4()))
    wandb.run.save()

@torch.no_grad()
def eval_actor(env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int) -> np.ndarray:
    """a\x8bδ   Ɲ  ˬ   ˡ       ͩ͢ ϧ"""
    env.seed(seed)
    actor.eval()
    EPISODE_REWARDS = []
    for _ in range(n_episodes):
        (state, done) = (env.reset(), False)
        episode_reward = 0.0
        while not done:
            act_ion = actor.act(state, device)
            (state, rewa_rd, done, _) = env.step(act_ion)
            episode_reward += rewa_rd
        EPISODE_REWARDS.append(episode_reward)
    actor.train()
    return np.asarray(EPISODE_REWARDS)

def return_reward_range(datasetnlJ, max_episo):
    (returns, l) = ([], [])
    (ep_ret, ep_len) = (0.0, 0)
    for (r, d) in zip(datasetnlJ['rewards'], datasetnlJ['terminals']):
        ep_ret += flo_at(r)
        ep_len += 1
        if d or ep_len == max_episo:
            returns.append(ep_ret)
            l.append(ep_len)
            (ep_ret, ep_len) = (0.0, 0)
    l.append(ep_len)
    assert sum(l) == len(datasetnlJ['rewards'])
    return (min(returns), ma(returns))

def modify_reward(datasetnlJ, env_n_ame, max_episo=1000):
    """  Ʊ ŕ  ˧   ϭ Ƶ """
    if aO((s_ in env_n_ame for s_ in ('halfcheetah', 'hopper', 'walker2d'))):
        (min__ret, max_ret) = return_reward_range(datasetnlJ, max_episo)
        datasetnlJ['rewards'] /= max_ret - min__ret
        datasetnlJ['rewards'] *= max_episo
    elif 'antmaze' in env_n_ame:
        datasetnlJ['rewards'] -= 1.0

class Actor(nn.Module):
    """  ̠϶̾   ˓  """

    def __init__(self, state_dime: int, action_dim: int, max_action: flo_at):
        sup_er(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dime, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str='cpu') -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

class CRITIC(nn.Module):
    """ Ÿ\x82͖ȸ ̕   Ʊ̏\x92     ϋ  ˪ ϣϖ ʹ   """

    def forward(self, state: torch.Tensor, act_ion: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, act_ion], 1)
        return self.net(sa)

    def __init__(self, state_dime: int, action_dim: int):
        sup_er(CRITIC, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dime + action_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

class TD3_BC:
    """   \x8bϲ  ú hè"""

    def train(self, batch_: TensorBatch) -> Dict[str, flo_at]:
        """ \u0381   ψ \x9d Ϫ̈́    Ŀ ¹uȕ̂ĕ˧ """
        log_d = {}
        self.total_it += 1
        (state, act_ion, rewa_rd, next_state, done) = batch_
        not_done = 1 - done
        with torch.no_grad():
            _noise = (torch.randn_like(act_ion) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + _noise).clamp(-self.max_action, self.max_action)
            target_q1 = self.critic_1_target(next_state, next_action)
            tm = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, tm)
            target_q = rewa_rd + not_done * self.discount * target_q
        current_q1 = self.critic_1(state, act_ion)
        current_q2 = self.critic_2(state, act_ion)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_d['critic_loss'] = critic_loss.item()
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        if self.total_it % self.policy_freq == 0:
            piU = self.actor(state)
            q = self.critic_1(state, piU)
            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + F.mse_loss(piU, act_ion)
            log_d['actor_loss'] = actor_loss.item()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)
        return log_d

    def state_dict(self) -> Dict[str, Any]:
        """Ƙ              """
        return {'critic_1': self.critic_1.state_dict(), 'critic_1_optimizer': self.critic_1_optimizer.state_dict(), 'critic_2': self.critic_2.state_dict(), 'critic_2_optimizer': self.critic_2_optimizer.state_dict(), 'actor': self.actor.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'total_it': self.total_it}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict['critic_1'])
        self.critic_1_optimizer.load_state_dict(state_dict['critic_1_optimizer'])
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2.load_state_dict(state_dict['critic_2'])
        self.critic_2_optimizer.load_state_dict(state_dict['critic_2_optimizer'])
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)
        self.total_it = state_dict['total_it']

    def __init__(self, max_action: flo_at, actor: nn.Module, actor_optimizer: torch.optim.Optimizer, critic_1: nn.Module, critic_1_optimizer: torch.optim.Optimizer, critic_2: nn.Module, critic_2_optimizer: torch.optim.Optimizer, discount: flo_at=0.99, tau: flo_at=0.005, policy_noise: flo_at=0.2, noise_clip: flo_at=0.5, policy_freq: int=2, a: flo_at=2.5, device: str='cpu'):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = a
        self.total_it = 0
        self.device = device

@pyrallis.wrap()
def train(config: TrainConfig):
    """ľ Ǧ  ̢  """
    env = gym.make(config.env)
    state_dime = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    datasetnlJ = d4rl.qlearning_dataset(env)
    if config.normalize_reward:
        modify_reward(datasetnlJ, config.env)
    if config.normalize:
        (state_mean, state_std) = compute_mean_stdOm(datasetnlJ['observations'], eps=0.001)
    else:
        (state_mean, state_std) = (0, 1)
    datasetnlJ['observations'] = NORMALIZE_STATES(datasetnlJ['observations'], state_mean, state_std)
    datasetnlJ['next_observations'] = NORMALIZE_STATES(datasetnlJ['next_observations'], state_mean, state_std)
    env = wrap_(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(state_dime, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(datasetnlJ)
    max_action = flo_at(env.action_space.high[0])
    if config.checkpoints_path is not None:
        print(f'Checkpoints path: {config.checkpoints_path}')
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, 'config.yaml'), 'w') as f:
            pyrallis.dump(config, f)
    seed = config.seed
    set_seed_(seed, env)
    actor = Actor(state_dime, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    critic_1 = CRITIC(state_dime, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=0.0003)
    critic_2 = CRITIC(state_dime, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=0.0003)
    kwargs = {'max_action': max_action, 'actor': actor, 'actor_optimizer': actor_optimizer, 'critic_1': critic_1, 'critic_1_optimizer': critic_1_optimizer, 'critic_2': critic_2, 'critic_2_optimizer': critic_2_optimizer, 'discount': config.discount, 'tau': config.tau, 'device': config.device, 'policy_noise': config.policy_noise * max_action, 'noise_clip': config.noise_clip * max_action, 'policy_freq': config.policy_freq, 'alpha': config.alpha}
    print('---------------------------------------')
    print(f'Training TD3 + BC, Env: {config.env}, Seed: {seed}')
    print('---------------------------------------')
    t_rainer = TD3_BC(**kwargs)
    if config.load_model != '':
        policy_file = Path(config.load_model)
        t_rainer.load_state_dict(torch.load(policy_file))
        actor = t_rainer.actor
    wandb_init(asdict(config))
    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch_ = replay_buffer.sample(config.batch_size)
        batch_ = [b.to(config.device) for b in batch_]
        log_d = t_rainer.train(batch_)
        wandb.log(log_d, step=t_rainer.total_it)
        if (t + 1) % config.eval_freq == 0:
            print(f'Time steps: {t + 1}')
            EVAL_SCORES = eval_actor(env, actor, device=config.device, n_episodes=config.n_episodes, seed=config.seed)
            eval_score = EVAL_SCORES.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print('---------------------------------------')
            print(f'Evaluation over {config.n_episodes} episodes: {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}')
            print('---------------------------------------')
            torch.save(t_rainer.state_dict(), os.path.join(config.checkpoints_path, f'checkpoint_{t}.pt'))
            wandb.log({'d4rl_normalized_score': normalized_eval_score}, step=t_rainer.total_it)
if __name__ == '__main__':
    train()
