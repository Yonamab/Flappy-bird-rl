import sys
import random

import pygame
import numpy as np

from flappybird_rl.config import (
    WIDTH,
    HEIGHT,
    FPS,
    BIRD_X,
    BIRD_SIZE,
    GRAVITY,
    JUMP_STRENGTH,
    PIPE_WIDTH,
    PIPE_GAP,
    PIPE_SPEED,
    PIPE_DISTANCE,
    BLACK,
    BLUE,
    GREEN,
    RED,
)


class FlappyBirdEnv:
    """
    This is the reinforcement learning environment.

    The AI interacts with this class using:

    reset()
        Starts a new game and returns the first state.

    step(action)
        Applies an action and returns:
        next_state, reward, done, score

    Actions:
        0 = do nothing
        1 = flap

    State:
        [
            bird_y,
            bird_velocity,
            pipe_x_distance,
            pipe_gap_center_y,
            bird_distance_from_gap
        ]
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
        bird_distance_from_gap_normalized = (
            bird_center_y - next_pipe["gap_center"]
        ) / HEIGHT

        return np.array(
            [
                bird_y_normalized,
                velocity_normalized,
                pipe_x_distance_normalized,
                gap_center_normalized,
                bird_distance_from_gap_normalized,
            ],
            dtype=np.float32,
        )

    def step(self, action):
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

            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(
                bottom_pipe_rect
            ):
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