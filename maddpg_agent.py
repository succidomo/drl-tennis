
import random
import copy
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import your existing Actor network from model.py
# (Keep Actor decentralized; add centralized critic here)
from model import Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparameters (reasonable Tennis defaults)
# ----------------------------
BUFFER_SIZE     = int(1e6)   # replay buffer size
BATCH_SIZE      = 256        # minibatch size
GAMMA           = 0.99       # discount factor
TAU             = 1e-3       # soft update of target network parameters
LR_ACTOR        = 1e-4       # actor learning rate
LR_CRITIC       = 1e-3       # critic learning rate
WEIGHT_DECAY    = 0.0        # L2 for critic (set small >0 if unstable)
CRITIC_CLIP_NORM= 1.0        # gradient clipping for critic
UPDATE_EVERY    = 1          # learn every n environment steps
UPDATES_PER_STEP= 1          # number of gradient steps per env step (increase once buffer is warm)

# Ornstein-Uhlenbeck noise parameters
OU_MU           = 0.0
OU_THETA        = 0.15
OU_SIGMA_START  = 0.2        # initial exploration
OU_SIGMA_END    = 0.05       # min exploration
OU_SIGMA_DECAY  = 1e-6       # per step decay (tune as needed)


# ----------------------------
# Utilities
# ----------------------------
def hidden_init(layer: nn.Linear):
    # fan_in = in_features
    fan_in = layer.weight.data.size()[1]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CentralCritic(nn.Module):
    """Centralized Critic Q(o1,o2,a1,a2) for two agents."""
    def __init__(self, state_size, action_size, seed, hidden1=400, hidden2=300, num_agents=2):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.num_agents = num_agents
        obs_dim = num_agents * state_size
        act_dim = num_agents * action_size

        self.fcs1 = nn.Linear(obs_dim, hidden1)
        self.fc2  = nn.Linear(hidden1 + act_dim, hidden2)
        self.fc3  = nn.Linear(hidden2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        for m in [self.fcs1, self.fc2, self.fc3]:
            nn.init.zeros_(m.bias)

    def forward(self, obs_all, act_all):
        """obs_all: (B, 2*state_size), act_all: (B, 2*action_size)"""
        x = F.relu(self.fcs1(obs_all))
        x = torch.cat([x, act_all], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process. Per-actor (shape = (action_size,))."""
    def __init__(self, size, seed, mu=OU_MU, theta=OU_THETA, sigma=OU_SIGMA_START):
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.sigma_start = sigma
        self.sigma_end = OU_SIGMA_END
        self.sigma_decay = OU_SIGMA_DECAY
        self.state = None
        self.reset()
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.state = np.copy(self.mu)

    def decay_sigma(self):
        self.sigma = max(self.sigma_end, self.sigma - self.sigma_decay)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(size=x.shape)
        self.state = x + dx
        return self.state.astype(np.float32)


class ReplayBuffer:
    """Fixed-size buffer to store joint experience tuples for 2 agents."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["obs_all", "act_all", "rew_all", "next_obs_all", "done_all"])
        random.seed(seed)
        np.random.seed(seed)

    def add(self, obs_all, act_all, rew_all, next_obs_all, done_all):
        """Store 1 transition.
        obs_all, next_obs_all: shape (2*S,)
        act_all: shape (2*A,)
        rew_all, done_all: shape (2,)
        """
        e = self.experience(obs_all, act_all, rew_all, next_obs_all, done_all)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        obs_all       = torch.from_numpy(np.vstack([e.obs_all for e in experiences])).float().to(device)          # (B, 2*S)
        act_all       = torch.from_numpy(np.vstack([e.act_all for e in experiences])).float().to(device)          # (B, 2*A)
        rew_all       = torch.from_numpy(np.vstack([e.rew_all for e in experiences])).float().to(device)          # (B, 2)
        next_obs_all  = torch.from_numpy(np.vstack([e.next_obs_all for e in experiences])).float().to(device)     # (B, 2*S)
        done_all      = torch.from_numpy(np.vstack([e.done_all for e in experiences]).astype(np.uint8)).float().to(device)  # (B, 2)

        return (obs_all, act_all, rew_all, next_obs_all, done_all)

    def __len__(self):
        return len(self.memory)


class MADDPG:
    """Two decentralized actors with a single centralized critic.
       - actors see their own obs only
       - critic sees both obs and both actions
    """
    def __init__(self, state_size, action_size, num_agents=2, random_seed=0):
        assert num_agents == 2, "This minimal MADDPG implementation expects exactly 2 agents."
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Actors (local + target) for each agent
        self.actors_local  = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actors_opt    = [optim.Adam(self.actors_local[i].parameters(), lr=LR_ACTOR) for i in range(num_agents)]

        # Centralized Critic (local + target)
        self.critic_local  = CentralCritic(state_size, action_size, random_seed, num_agents=num_agents).to(device)
        self.critic_target = CentralCritic(state_size, action_size, random_seed, num_agents=num_agents).to(device)
        self.critic_opt    = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Initialize targets
        for i in range(num_agents):
            self._hard_update(self.actors_target[i], self.actors_local[i])
        self._hard_update(self.critic_target, self.critic_local)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        # OU noise per actor
        self.noise = [OUNoise(action_size, random_seed) for _ in range(num_agents)]
        self.t_step = 0

    # --------------- Act / Step ---------------
    def act(self, states, add_noise=True):
        """Compute actions for both agents given their observations.
        states: np.ndarray of shape (2, state_size)
        returns np.ndarray of shape (2, action_size), dtype float32 in [-1, 1]
        """
        self._eval_mode(self.actors_local)
        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32)).to(device)  # (2, S)
        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                a = self.actors_local[i](states_t[i]).cpu().data.numpy()
                if add_noise:
                    a = a + self.noise[i].sample()
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                actions.append(a)
        self._train_mode(self.actors_local)

        actions = np.vstack(actions).astype(np.float32)  # (2, A)
        return actions

    def reset(self):
        for n in self.noise:
            n.reset()

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay buffer, and learn when due.
        - states, next_states: (2, S)
        - actions: (2, A)
        - rewards, dones: (2,)
        """
        obs_all      = np.asarray(states, dtype=np.float32).reshape(-1)         # (2*S,)
        next_obs_all = np.asarray(next_states, dtype=np.float32).reshape(-1)    # (2*S,)
        act_all      = np.asarray(actions, dtype=np.float32).reshape(-1)        # (2*A,)
        rew_all      = np.asarray(rewards, dtype=np.float32).reshape(-1)        # (2,)
        done_all     = np.asarray(dones, dtype=np.uint8).reshape(-1)            # (2,)

        self.memory.add(obs_all, act_all, rew_all, next_obs_all, done_all)

        # Learn at fixed frequency
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                self.learn()

        # slowly decay exploration
        for n in self.noise:
            n.decay_sigma()

    # --------------- Learn ---------------
    def learn(self):
        obs_all, act_all, rew_all, next_obs_all, done_all = self.memory.sample()

        # Slice helpers
        def split_obs(t_obs):
            o1 = t_obs[:, :self.state_size]
            o2 = t_obs[:, self.state_size:]
            return o1, o2

        def split_act(t_act):
            a1 = t_act[:, :self.action_size]
            a2 = t_act[:, self.action_size:]
            return a1, a2

        # Next actions from target actors
        next_o1, next_o2 = split_obs(next_obs_all)
        a1_next = self.actors_target[0](next_o1)
        a2_next = self.actors_target[1](next_o2)
        a_all_next = torch.cat([a1_next, a2_next], dim=1)

        # Critic target
        # Tennis typically gives identical rewards to both agents; we use mean reward
        r = rew_all.mean(dim=1, keepdim=True)                       # (B,1)
        d, _ = done_all.max(dim=1, keepdim=True)                # (B,1)
        q_next = self.critic_target(next_obs_all, a_all_next)
        q_target = r + GAMMA * (1.0 - d) * q_next

        # Critic loss & update
        q_expected = self.critic_local(obs_all, act_all)
        critic_loss = F.mse_loss(q_expected, q_target.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        if CRITIC_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), CRITIC_CLIP_NORM)
        self.critic_opt.step()

        # Actor 1: maximize Q by choosing a1; keep a2 fixed (stop-grad)
        o1, o2 = split_obs(obs_all)
        a1 = self.actors_local[0](o1)
        with torch.no_grad():
            a2_fixed = self.actors_local[1](o2)
        a_all_1 = torch.cat([a1, a2_fixed], dim=1)
        actor1_loss = -self.critic_local(obs_all, a_all_1).mean()
        self.actors_opt[0].zero_grad()
        actor1_loss.backward()
        self.actors_opt[0].step()

        # Actor 2: symmetric
        with torch.no_grad():
            a1_fixed = self.actors_local[0](o1)
        a2 = self.actors_local[1](o2)
        a_all_2 = torch.cat([a1_fixed, a2], dim=1)
        actor2_loss = -self.critic_local(obs_all, a_all_2).mean()
        self.actors_opt[1].zero_grad()
        actor2_loss.backward()
        self.actors_opt[1].step()

        # Soft-update targets
        for i in range(self.num_agents):
            self._soft_update(self.actors_local[i], self.actors_target[i], TAU)
        self._soft_update(self.critic_local, self.critic_target, TAU)

    # --------------- Utils ---------------
    @staticmethod
    def _soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

    @staticmethod
    def _eval_mode(modules):
        if isinstance(modules, (list, tuple)):
            for m in modules:
                m.eval()
        else:
            modules.eval()

    @staticmethod
    def _train_mode(modules):
        if isinstance(modules, (list, tuple)):
            for m in modules:
                m.train()
        else:
            modules.train()
