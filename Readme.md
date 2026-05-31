# Flappy Bird Reinforcement Learning

This project builds a simple Flappy Bird game using Pygame and trains an AI agent to play it using Deep Q-Learning with PyTorch.

## How to Install

pip install -r requirements.txt

## How to Train

python flappy_dqn.py train

## How to Watch the Trained Agent

python flappy_dqn.py play

## Libraries Used

- Pygame
- NumPy
- PyTorch

## Reinforcement Learning Idea

The agent observes the bird position, bird velocity, distance to the next pipe, and pipe gap position.  
It chooses between two actions: do nothing or flap.  
The neural network learns which action gives the best long-term reward.