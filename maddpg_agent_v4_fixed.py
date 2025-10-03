
import random
import copy
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import DDPG_Models  # uses your existing Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Hyperparameters (simplified, conservative)
# ----------------------------
BUFFER_SIZE      = 10000
BATCH_SIZE       = 256
GAMMA            = 0.99
TAU              = 1e-3        # slow target updates
LR_ACTOR         = 1e-4
LR_CRITIC        = 1e-3
WEIGHT_DECAY     = 0.0
CRITIC_CLIP_NORM = 1.0
ACTOR_CLIP_NORM  = None

UPDATE_EVERY     = 2           # learn once every 2 env steps
UPDATES_PER_STEP = 1           # one gradient step per learn
POLICY_DELAY     = 1           # no delay (simpler)

WARMUP_STEPS     = 10000       # collect diverse data first
LEARN_AFTER_STEPS= 5000

# OU noise (keep exploration alive; no decay by default)
OU_MU            = 0.0
OU_THETA         = 0.15
OU_SIGMA_START   = 0.2
OU_SIGMA_END     = 0.2         # no decay (constant)
OU_SIGMA_DECAY   = 0.0

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

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
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.obs_all for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.act_all for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rew_all for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_obs_all for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done_all for e in experiences]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
class DDPG:
    def __init__(self, agent_id, model, action_size, random_seed):
        self.id = agent_id
        
        #actor models
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #critic models
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Hard update targets
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        self.noise = OUNoise(action_size, random_seed)

    
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self, state, numepisode, noise_weight=1.0, add_noise=True):
        # Similar to your act(), but per-agent (no loop)
        state = torch.from_numpy(state).float().to(device)  # (24,)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            #self.noise_val = self.noise.sample() * noise_weight           
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
            #action += (self.noise.sample()/np.sqrt(numepisode))  
        return np.clip(action, -1.0, 1.0)

    def learn(self, agent_index, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences

        self.critic_opt.zero_grad()
        agent_id = torch.tensor([agent_index]).to(device)

        # Critic update
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_tnext = self.critic_target(next_states, actions_next)

        q_target = rewards.index_select(1, agent_id) + gamma * (1 - dones.index_select(1, agent_id)) * q_tnext
        q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expected, q_target.detach())
        critic_loss.backward()

        self.critic_opt.step()

        # Actor update
        self.actor_opt.zero_grad()
        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()

        self.actor_opt.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    
        

class MADDPG:
    """Two decentralized actors + per-agent centralized critics (simplified)."""
    def __init__(self, state_size, action_size, num_agents=2, random_seed=0, noise_decay=1.0, t_stop_noise=30000, noise_start=1.0):
        #assert num_agents == 2, "This implementation is specialized for 2 agents."
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.n_agents = num_agents
        self.noise_on = True
        self.noise_decay = noise_decay
        self.t_stop_noise = t_stop_noise
        self.noise_weight = noise_start

        random.seed(random_seed); np.random.seed(random_seed); torch.manual_seed(random_seed)

        models = [DDPG_Models(num_agents) for _ in range(num_agents)]
        self.agents = [DDPG(i, models[i], action_size, random_seed) for i in range(num_agents)]

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.noise  = [OUNoise(action_size, random_seed) for _ in range(num_agents)]

        self.t_step = 0
        self.total_steps = 0
        self._learn_count = 0

    def reset(self):
        for n in self.noise:
            n.reset()

    def act(self, states, numepisode, add_noise=True, training=True):
        # pass each agents state from the env and calculate its action

        all_actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act( state, numepisode, self.noise_weight, self.noise_on )
            self.noise_weight *= self.noise_decay
            all_actions.append(action)

        return np.array(all_actions).reshape(1, -1)
        

    def step(self, states, actions, rewards, next_states, dones):
        # flatten obs/actions; keep per-agent reward/done as len-2
        obs_all = states.reshape(1, -1)
        next_obs_all = next_states.reshape(1, -1)
        #act_all = actions.reshape(1, -1)              # not sure about flatten here

        self.memory.add(obs_all, actions, rewards, next_obs_all, dones)
        self.total_steps += 1

        if self.t_step > self.t_stop_noise:
            self.noise_on = False

        # learn on schedule
        self.t_step = (self.t_step + 1) # % UPDATE_EVERY

        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > self.batch_size:
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        all_next_actions = []
        all_actions = []

        for i, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # get agent state and actions
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

        for i, agent in enumerate(self.agents):
            agent.learn( i, experiences[i], GAMMA, all_next_actions, all_actions)
 
            

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


