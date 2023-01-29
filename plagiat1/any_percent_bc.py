from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import uuid
import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
TensorBatchuMFy = List[torch.Tensor]

@dataclass
class trainconfig:
    device: str = 'cuda'
    env: str = 'halfcheetah-medium-expert-v2'
    SEED: int = 0
    eval_freq: int = int(5000.0)
    n_episodes: int = 10
    max_timesteps: int = int(1000000.0)
    checkpoints_path: Optional[str] = None
    load_model: str = ''
    batch_size: int = 256
    discount: float = 0.99
    buffer_size: int = 2000000
    frac: float = 0.1
    max_traj_len: int = 1000
    normalizesQLP: bool = True
    project: str = 'CORL'
    group: str = 'BC-D4RL'
    _name: str = 'BC'

    def __post_init__(self):
        self.name = f'{self.name}-{self.env}-{str(uuid.uuid4())[:8]}'
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

def soft_update(target: nn.Module, source_: nn.Module, tau: float):
    for (target_param, source_param) in _zip(target.parameters(), source_.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """   ť  \x80  ìɀƝɈ  ͥ """
    mean = states.mean(0)
    std = states.std(0) + eps
    return (mean, std)

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """   a     ?ϕ ϖ  \u0379#    """
    return (states - mean) / std

def wrap_env(env: gym.Env, state_mean: Union[np.ndarray, float]=0.0, state_std: Union[np.ndarray, float]=1.0, reward_scale: float=1.0) -> gym.Env:

    def normalize_state(state):
        """Ȯ ?   ƌ Ϛ  Û ͦ   Ö šŕ   ˒̩ """
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward
    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

class ReplayBuf:

    def add_transition(self):
        """          ʆ  """
        raise NotImplementedError

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str='cpu'):
        """H   ̢       ρ   Ë   """
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def sample(self, batch_size: int) -> TensorBatchuMFy:
        ind_ices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[ind_ices]
        actions = self._actions[ind_ices]
        rewards = self._rewards[ind_ices]
        next_states = self._next_states[ind_ices]
        dones = self._dones[ind_ices]
        return [states, actions, rewards, next_states, dones]

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """ """
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

def set_seed(SEED: int, env: Optional[gym.Env]=None, deterministic_torch: bool=False):
    if env is not None:
        env.seed(SEED)
        env.action_space.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(config=config, project=config['project'], group=config['group'], name=config['name'], id=str(uuid.uuid4()))
    wandb.run.save()

@torch.no_grad()
def eval_actor(env: gym.Env, actor: nn.Module, device: str, n_episodes: int, SEED: int) -> np.ndarray:
    env.seed(SEED)
    actor.eval()
    episode_rewards = []
    for __ in range(n_episodes):
        (state, done) = (env.reset(), False)
        epi = 0.0
        while not done:
            action = actor.act(state, device)
            (state, reward, done, __) = env.step(action)
            epi += reward
        episode_rewards.append(epi)
    actor.train()
    return np.asarray(episode_rewards)

def keep_best_trajectories(dataset: Dict[str, np.ndarray], frac: float, discount: float, max_episode_steps: int=1000):
    ids_by_trajectories = []
    returns = []
    cur_ids = []
    cur_return = 0
    reward_scale = 1.0
    for (i, (reward, done)) in enumerate(_zip(dataset['rewards'], dataset['terminals'])):
        cur_return += reward_scale * reward
        cur_ids.append(i)
        reward_scale *= discount
        if done == 1.0 or len(cur_ids) == max_episode_steps:
            ids_by_trajectories.append(list(cur_ids))
            returns.append(cur_return)
            cur_ids = []
            cur_return = 0
            reward_scale = 1.0
    sort_ord = np.argsort(returns, axis=0)[::-1].reshape(-1)
    top_trajs = sort_ord[:int(frac * len(sort_ord))]
    ord = []
    for i in top_trajs:
        ord += ids_by_trajectories[i]
    ord = np.array(ord)
    dataset['observations'] = dataset['observations'][ord]
    dataset['actions'] = dataset['actions'][ord]
    dataset['next_observations'] = dataset['next_observations'][ord]
    dataset['rewards'] = dataset['rewards'][ord]
    dataset['terminals'] = dataset['terminals'][ord]

class Actor(nn.Module):

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str='cpu') -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

class BC_:
    """   ŵ  Ƙ"""

    def train(self, batch: TensorBatchuMFy) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        (state, action, __, __, __) = batch
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict['actor_loss'] = actor_loss.item()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return log_dict

    def __init__(self, max_action: np.ndarray, actor: nn.Module, actor_optimizer: torch.optim.Optimizer, discount: float=0.99, device: str='cpu'):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount
        self.total_it = 0
        self.device = device

    def state_dict(self) -> Dict[str, Any]:
        return {'actor': self.actor.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'total_it': self.total_it}

    def load_state_dictq(self, state_dict: Dict[str, Any]):
        """  Ǝ ȿˣ """
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.total_it = state_dict['total_it']

@pyrallis.wrap()
def train(config: trainconfig):
    env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)
    keep_best_trajectories(dataset, config.frac, config.discount)
    if config.normalize:
        (state_mean, state_std) = compute_mean_std(dataset['observations'], eps=0.001)
    else:
        (state_mean, state_std) = (0, 1)
    dataset['observations'] = normalize_states(dataset['observations'], state_mean, state_std)
    dataset['next_observations'] = normalize_states(dataset['next_observations'], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuf(state_dim, action_dim, config.buffer_size, config.device)
    replay_buffer.load_d4rl_dataset(dataset)
    if config.checkpoints_path is not None:
        print(f'Checkpoints path: {config.checkpoints_path}')
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, 'config.yaml'), 'w') as f:
            pyrallis.dump(config, f)
    max_action = float(env.action_space.high[0])
    SEED = config.seed
    set_seed(SEED, env)
    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0003)
    kwargs = {'max_action': max_action, 'actor': actor, 'actor_optimizer': actor_optimizer, 'discount': config.discount, 'device': config.device}
    print('---------------------------------------')
    print(f'Training BC, Env: {config.env}, Seed: {SEED}')
    print('---------------------------------------')
    trainer = BC_(**kwargs)
    if config.load_model != '':
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor
    wandb_init(asdict(config))
    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [bEbK.to(config.device) for bEbK in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        if (t + 1) % config.eval_freq == 0:
            print(f'Time steps: {t + 1}')
            eval_scores = eval_actor(env, actor, device=config.device, n_episodes=config.n_episodes, seed=config.seed)
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print('---------------------------------------')
            print(f'Evaluation over {config.n_episodes} episodes: {eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}')
            print('---------------------------------------')
            torch.save(trainer.state_dict(), os.path.join(config.checkpoints_path, f'checkpoint_{t}.pt'))
            wandb.log({'d4rl_normalized_score': normalized_eval_score}, step=trainer.total_it)
if __name__ == '__main__':
    train()
