import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flappybird_rl.config import MODEL_PATH
from flappybird_rl.model import DQN


class ReplayMemory:
    """
    Stores past experiences.

    Each experience contains:
        state, action, reward, next_state, done
    """

    def __init__(self, capacity=50_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.loss_fn = nn.MSELoss()

        self.gamma = 0.99
        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.999

        self.target_update_frequency = 10

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.

        Sometimes the agent explores by choosing randomly.
        Other times it exploits by choosing the best predicted action.
        """

        if random.random() < self.epsilon:
            return random.randint(0, 1)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            best_next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * best_next_q_values * (1 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save(self, path=MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path=MODEL_PATH):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # During play/evaluation, we do not want random exploratory actions.
        self.epsilon = 0.0