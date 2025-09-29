
import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor  # your existing Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparameters
# ----------------------------
BUFFER_SIZE      = int(1e6)   # Keep
BATCH_SIZE       = 128        # Keep
GAMMA            = 0.99       # Keep
TAU              = 1e-3       # Keep
LR_ACTOR         = 1e-4       # Revert to lower (was 1e-3; too high caused collapse)
LR_CRITIC        = 1e-3       # Keep
WEIGHT_DECAY     = 0          # Keep
CRITIC_CLIP_NORM = 1.0        # Re-enable mild clipping for stability
ACTOR_CLIP_NORM  = None       # Keep off

UPDATE_EVERY     = 2          # Revert to frequent small updates (was 20)
POLICY_DELAY     = 2          # Keep
UPDATES_PER_STEP = 2          # Reduce bursts (was 10; prevents overfitting)

WARMUP_STEPS     = 2000       # Slight increase for better buffer diversity
LEARN_AFTER_STEPS= 1000       # Keep

# OU noise with decay re-enabled
OU_MU            = 0.0
OU_THETA         = 0.15
OU_SIGMA_START   = 0.2
OU_SIGMA_END     = 0.0        # Decay to 0 for exploitation
OU_SIGMA_DECAY   = 1e-6       # Slower decay per step

def hidden_init(layer: nn.Linear):
    fan_in = layer.weight.data.size()[1]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CentralCritic(nn.Module):
    """Centralized critic Q_i(o1,o2,a1,a2) -> scalar (for agent i)."""
    def __init__(self, state_size, action_size, seed, hidden1=400, hidden2=300, num_agents=2):
        super().__init__()
        torch.manual_seed(seed)
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
        x = F.relu(self.fcs1(obs_all))
        x = torch.cat([x, act_all], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class OUNoise:
    """Per-actor OU noise (shape = (action_size,))."""
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
    """Replay buffer storing joint transitions for 2 agents."""
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
            field_names=["obs_all", "act_all", "rew_all", "next_obs_all", "done_all"])
        random.seed(seed)
        np.random.seed(seed)

    def add(self, obs_all, act_all, rew_all, next_obs_all, done_all):
        e = self.experience(obs_all, act_all, rew_all, next_obs_all, done_all)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        obs_all      = torch.from_numpy(np.vstack([e.obs_all for e in experiences])).float().to(device)
        act_all      = torch.from_numpy(np.vstack([e.act_all for e in experiences])).float().to(device)
        rew_all      = torch.from_numpy(np.vstack([e.rew_all for e in experiences])).float().to(device)      # (B,2)
        next_obs_all = torch.from_numpy(np.vstack([e.next_obs_all for e in experiences])).float().to(device)
        done_all     = torch.from_numpy(np.vstack([e.done_all for e in experiences]).astype(np.uint8)).float().to(device)
        return obs_all, act_all, rew_all, next_obs_all, done_all

    def __len__(self):
        return len(self.memory)

class MADDPG:
    """True MADDPG with per-agent centralized critics."""
    def __init__(self, state_size, action_size, num_agents=2, random_seed=0):
        assert num_agents == 2, "This implementation is specialized for 2 agents."
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        random.seed(random_seed); np.random.seed(random_seed); torch.manual_seed(random_seed)

        # Two actors (local & target)
        self.actors_local  = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actors_opt    = [optim.Adam(self.actors_local[i].parameters(), lr=LR_ACTOR) for i in range(num_agents)]

        # Two centralized critics (local & target), one per agent
        self.critics_local  = [CentralCritic(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critics_target = [CentralCritic(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.critics_opt    = [optim.Adam(self.critics_local[i].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
                                for i in range(num_agents)]

        # Sync targets
        for i in range(num_agents):
            self._hard_update(self.actors_target[i],  self.actors_local[i])
            self._hard_update(self.critics_target[i], self.critics_local[i])

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.noise  = [OUNoise(action_size, random_seed) for _ in range(num_agents)]

        self.t_step = 0
        self.total_steps = 0
        self._update_count = 0  # for policy delay

    def reset(self):
        for n in self.noise:
            n.reset()

    def act(self, states, add_noise=True, training=True):
        """states: (2, state_size). Returns (2, action_size) in [-1,1]."""
        states = np.asarray(states, dtype=np.float32)
        if training and self.total_steps < WARMUP_STEPS:
            # Pure random during warmup for diverse buffer
            return np.random.uniform(-1.0, 1.0, size=(self.num_agents, self.action_size)).astype(np.float32)

        actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                s = torch.from_numpy(states[i]).float().to(device)
                a = self.actors_local[i](s).cpu().data.numpy()
                if training and add_noise:
                    a = a + self.noise[i].sample()
                a = np.clip(a, -1.0, 1.0).astype(np.float32)
                actions.append(a)
        return np.vstack(actions).astype(np.float32)

    def step(self, states, actions, rewards, next_states, dones):
        # Store transition (flatten obs/actions; keep rewards/dones as length-2 vectors)
        obs_all      = np.asarray(states, dtype=np.float32).reshape(-1)
        next_obs_all = np.asarray(next_states, dtype=np.float32).reshape(-1)
        act_all      = np.asarray(actions, dtype=np.float32).reshape(-1)
        rew_all      = np.asarray(rewards, dtype=np.float32).reshape(-1)
        done_all     = np.asarray(dones, dtype=np.uint8).reshape(-1)

        self.memory.add(obs_all, act_all, rew_all, next_obs_all, done_all)
        self.total_steps += 1

        # Decay noise each env step
        for n in self.noise:
            n.decay_sigma()

        # Learn on schedule
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE and self.total_steps > LEARN_AFTER_STEPS:
            for _ in range(UPDATES_PER_STEP):
                self.learn()

    def learn(self):
        obs_all, act_all, rew_all, next_obs_all, done_all = self.memory.sample()

        self._learn_count = getattr(self, "_learn_count", 0) + 1

        # Slice obs
        o1 = obs_all[:, :self.state_size]
        o2 = obs_all[:, self.state_size:]
        next_o1 = next_obs_all[:, :self.state_size]
        next_o2 = next_obs_all[:, self.state_size:]

        # Target actions
        a1_next = self.actors_target[0](next_o1)
        a2_next = self.actors_target[1](next_o2)
        with torch.no_grad():
            n1 = (torch.randn_like(a1_next) * 0.2).clamp(-0.5, 0.5)
            n2 = (torch.randn_like(a2_next) * 0.2).clamp(-0.5, 0.5)
            a1_next = (a1_next + n1).clamp(-1, 1)
            a2_next = (a2_next + n2).clamp(-1, 1)
        a_all_next = torch.cat([a1_next, a2_next], dim=1)

        # Per-agent rewards/dones (column vectors)
        r1 = rew_all[:, 0:1]
        r2 = rew_all[:, 1:2]
        d1 = done_all[:, 0:1]
        d2 = done_all[:, 1:2]

        # ----- Critic updates (for each agent) -----
        q1_next = self.critics_target[0](next_obs_all, a_all_next)
        q2_next = self.critics_target[1](next_obs_all, a_all_next)

        q1_target = r1 + GAMMA * (1. - d1) * q1_next
        q2_target = r2 + GAMMA * (1. - d2) * q2_next

        q1_expected = self.critics_local[0](obs_all, act_all)
        q2_expected = self.critics_local[1](obs_all, act_all)

        critic1_loss = F.mse_loss(q1_expected, q1_target.detach())
        critic2_loss = F.mse_loss(q2_expected, q2_target.detach())

        self.critics_opt[0].zero_grad()
        critic1_loss.backward()
        if CRITIC_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.critics_local[0].parameters(), CRITIC_CLIP_NORM)
        self.critics_opt[0].step()

        if self._learn_count % 100 == 0:
            print(f"Learn step {self._learn_count}: Critic1 loss: {critic1_loss.item():.4f}, Critic2 loss: {critic2_loss.item():.4f}")

        self.critics_opt[1].zero_grad()
        critic2_loss.backward()
        if CRITIC_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.critics_local[1].parameters(), CRITIC_CLIP_NORM)
        self.critics_opt[1].step()

        # ----- Actor updates (delayed) -----
        self._update_count = getattr(self, "_update_count", 0) + 1
        if self._update_count % POLICY_DELAY == 0:
            a1 = self.actors_local[0](o1)
            a2 = self.actors_local[1](o2)

            # Stop-grad partner for each actor
            with torch.no_grad():
                a2_fixed = self.actors_local[1](o2)
                a1_fixed = self.actors_local[0](o1)

            # Actor 1 maximizes critic1
            a_all_1 = torch.cat([a1, a2_fixed], dim=1)
            actor1_loss = -self.critics_local[0](obs_all, a_all_1).mean() # + 1e-3 * (a1**2).mean()
            self.actors_opt[0].zero_grad()
            actor1_loss.backward()
            if ACTOR_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(self.actors_local[0].parameters(), ACTOR_CLIP_NORM)
            self.actors_opt[0].step()

            # Actor 2 maximizes critic2
            a_all_2 = torch.cat([a1_fixed, a2], dim=1)
            actor2_loss = -self.critics_local[1](obs_all, a_all_2).mean() # + 1e-3 * (a2**2).mean()
            self.actors_opt[1].zero_grad()
            actor2_loss.backward()
            if ACTOR_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(self.actors_local[1].parameters(), ACTOR_CLIP_NORM)
            self.actors_opt[1].step()

            if self._learn_count % 100 == 0:
                print(f"Learn step {self._learn_count}: Actor1 loss: {actor1_loss.item():.4f}, Actor2 loss: {actor2_loss.item():.4f}")

            # Soft-update targets
            for i in range(self.num_agents):
                self._soft_update(self.actors_local[i],  self.actors_target[i], TAU)
                self._soft_update(self.critics_local[i], self.critics_target[i], TAU)

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _hard_update(target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)
