"""
Flappy Bird Reinforcement Learning Project
Single-file version using Pygame + PyTorch DQN

Goal:
- Build a simple Flappy Bird environment manually.
- Train an AI agent using a Deep Q-Network.
- Keep the code understandable for a school project submission.

Install requirements:
    pip install pygame torch numpy

Run training:
    python flappy_dqn.py train

Watch trained agent:
    python flappy_dqn.py play
"""

import sys
import random
import os
from collections import deque

import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. BASIC GAME SETTINGS
# ============================================================

WIDTH = 400
HEIGHT = 600
FPS = 60

BIRD_X = 80
BIRD_SIZE = 28
GRAVITY = 0.5
JUMP_STRENGTH = -8.5

PIPE_WIDTH = 70
PIPE_GAP = 185
PIPE_SPEED = 3
PIPE_DISTANCE = 220

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (80, 180, 255)
GREEN = (80, 200, 120)
RED = (240, 80, 80)


# ============================================================
# 2. FLAPPY BIRD ENVIRONMENT
# ============================================================

class FlappyBirdEnv:
    """
    This class is the reinforcement learning environment.

    The agent does not directly control the game screen.
    Instead, it interacts with the environment using:

    - reset() -> starts a new game and returns the first state
    - step(action) -> applies an action and returns:
        next_state, reward, done, score

    Actions:
        0 = do nothing
        1 = flap / jump

    State given to the AI:
        [bird_y, bird_velocity, pipe_x_distance, pipe_gap_center_y, bird_distance_from_gap]

    These numbers describe the situation clearly enough for learning.
    """

    def __init__(self, render=False):
        self.render_mode = render
        self.screen = None
        self.clock = None
        self.font = None

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Flappy Bird DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

        self.reset()

    def reset(self):
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.frame_count = 0
        self.done = False

        self._add_pipe(WIDTH + 100)
        self._add_pipe(WIDTH + 100 + PIPE_DISTANCE)

        return self._get_state()

    def _add_pipe(self, x):
        gap_center = random.randint(140, HEIGHT - 140)
        pipe = {
            "x": x,
            "gap_center": gap_center,
            "passed": False,
        }
        self.pipes.append(pipe)

    def _get_next_pipe(self):
        for pipe in self.pipes:
            if pipe["x"] + PIPE_WIDTH >= BIRD_X:
                return pipe
        return self.pipes[0]

    def _get_state(self):
        next_pipe = self._get_next_pipe()

        bird_center_y = self.bird_y + BIRD_SIZE / 2

        bird_y_normalized = bird_center_y / HEIGHT
        velocity_normalized = self.bird_velocity / 10
        pipe_x_distance_normalized = (next_pipe["x"] - BIRD_X) / WIDTH
        gap_center_normalized = next_pipe["gap_center"] / HEIGHT
        bird_distance_from_gap_normalized = (bird_center_y - next_pipe["gap_center"]) / HEIGHT

        return np.array([
            bird_y_normalized,
            velocity_normalized,
            pipe_x_distance_normalized,
            gap_center_normalized,
            bird_distance_from_gap_normalized,
        ], dtype=np.float32)

    def step(self, action):
        """
        One game update.

        Reward design:
        - small positive reward every frame for staying alive
        - larger positive reward for passing a pipe
        - large negative reward for crashing
        """

        if action == 1:
            self.bird_velocity = JUMP_STRENGTH

        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe["x"] -= PIPE_SPEED

        if self.pipes[-1]["x"] < WIDTH - PIPE_DISTANCE:
            self._add_pipe(WIDTH)

        if self.pipes[0]["x"] + PIPE_WIDTH < 0:
            self.pipes.pop(0)

        reward = 0.05

        # Reward shaping: give a small bonus when the bird is near the center
        # of the next pipe gap. This helps early learning a lot.
        next_pipe = self._get_next_pipe()
        bird_center_y = self.bird_y + BIRD_SIZE / 2
        distance_from_gap = abs(bird_center_y - next_pipe["gap_center"])
        reward += max(0, 1 - distance_from_gap / 250) * 0.05

        for pipe in self.pipes:
            if not pipe["passed"] and pipe["x"] + PIPE_WIDTH < BIRD_X:
                pipe["passed"] = True
                self.score += 1
                reward += 5.0

        if self._check_collision():
            self.done = True
            reward = -10.0

        self.frame_count += 1

        if self.render_mode:
            self.render()

        return self._get_state(), reward, self.done, self.score

    def _check_collision(self):
        bird_rect = pygame.Rect(
            BIRD_X,
            int(self.bird_y),
            BIRD_SIZE,
            BIRD_SIZE,
        )

        if self.bird_y < 0 or self.bird_y + BIRD_SIZE > HEIGHT:
            return True

        for pipe in self.pipes:
            top_pipe_rect = pygame.Rect(
                pipe["x"],
                0,
                PIPE_WIDTH,
                pipe["gap_center"] - PIPE_GAP // 2,
            )
            bottom_pipe_rect = pygame.Rect(
                pipe["x"],
                pipe["gap_center"] + PIPE_GAP // 2,
                PIPE_WIDTH,
                HEIGHT,
            )

            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                return True

        return False

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(BLUE)

        pygame.draw.rect(
            self.screen,
            RED,
            pygame.Rect(BIRD_X, int(self.bird_y), BIRD_SIZE, BIRD_SIZE),
        )

        for pipe in self.pipes:
            top_height = pipe["gap_center"] - PIPE_GAP // 2
            bottom_y = pipe["gap_center"] + PIPE_GAP // 2

            pygame.draw.rect(
                self.screen,
                GREEN,
                pygame.Rect(pipe["x"], 0, PIPE_WIDTH, top_height),
            )
            pygame.draw.rect(
                self.screen,
                GREEN,
                pygame.Rect(pipe["x"], bottom_y, PIPE_WIDTH, HEIGHT - bottom_y),
            )

        score_text = self.font.render(f"Score: {self.score}", True, BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)


# ============================================================
# 3. NEURAL NETWORK
# ============================================================

class DQN(nn.Module):
    """
    Deep Q-Network.

    Input: 4 state values
    Output: 2 Q-values
        output[0] = predicted value of doing nothing
        output[1] = predicted value of flapping
    """

    def __init__(self, input_size=5, output_size=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 4. EXPERIENCE REPLAY MEMORY
# ============================================================

class ReplayMemory:
    """
    The AI stores past experiences here.

    Each experience contains:
        state, action, reward, next_state, done

    Instead of learning only from the latest moment,
    the agent learns from random past moments.
    This makes training more stable.
    """

    def __init__(self, capacity=50_000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ============================================================
# 5. DQN AGENT
# ============================================================

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
        Other times it exploits by choosing the action with the best Q-value.
        """

        if random.random() < self.epsilon:
            return random.randint(0, 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

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

    def save(self, path="models/flappy_dqn_model.pth"):
        os.makedirs("models", exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="models/flappy_dqn_model.pth"):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = 0.0


# ============================================================
# 6. TRAINING LOOP
# ============================================================

def train():
    env = FlappyBirdEnv(render=False)
    agent = DQNAgent()

    episodes = 5000
    best_score = 0

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        losses = []

        while True:
            action = agent.choose_action(state)
            next_state, reward, done, score = env.step(action)

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.learn()

            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            # Safety limit so one very long episode does not run forever.
            if done or env.frame_count > 5000:
                break

        agent.decay_epsilon()

        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()

        if score > best_score:
            best_score = score
            agent.save()

        average_loss = sum(losses) / len(losses) if losses else 0

        print(
            f"Episode {episode:4d} | "
            f"Score: {score:3d} | "
            f"Best: {best_score:3d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Loss: {average_loss:8.4f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    agent.save()
    print("Training complete. Model saved as flappy_dqn_model.pth")


# ============================================================
# 7. WATCH TRAINED AGENT
# ============================================================

def play():
    env = FlappyBirdEnv(render=True)
    agent = DQNAgent()

    try:
        agent.load()
    except FileNotFoundError:
        print("No trained model found. Run: python flappy_dqn.py train")
        return

    while True:
        state = env.reset()

        while True:
            action = agent.choose_action(state)
            state, reward, done, score = env.step(action)

            if done:
                pygame.time.wait(700)
                break


# ============================================================
# 8. MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please choose a mode:")
        print("  python flappy_dqn.py train")
        print("  python flappy_dqn.py play")
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "play":
        play()
    else:
        print("Unknown mode. Use train or play.")
