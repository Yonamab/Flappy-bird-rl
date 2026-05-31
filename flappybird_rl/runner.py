import pygame

from flappybird_rl.environment import FlappyBirdEnv
from flappybird_rl.agent import DQNAgent


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
    print("Training complete. Model saved as models/flappy_dqn_model.pth")


def play():
    env = FlappyBirdEnv(render=True)
    agent = DQNAgent()

    try:
        agent.load()
    except FileNotFoundError:
        print("No trained model found. Run: python main.py train")
        return

    while True:
        state = env.reset()

        while True:
            action = agent.choose_action(state)

            state, reward, done, score = env.step(action)

            if done:
                pygame.time.wait(700)
                break


def evaluate():
    env = FlappyBirdEnv(render=False)
    agent = DQNAgent()

    try:
        agent.load()
    except FileNotFoundError:
        print("No trained model found. Run: python main.py train")
        return

    episodes = 20
    scores = []

    for episode in range(1, episodes + 1):
        state = env.reset()

        while True:
            action = agent.choose_action(state)

            state, reward, done, score = env.step(action)

            if done or env.frame_count > 5000:
                break

        scores.append(score)

        print(f"Evaluation episode {episode:2d} | Score: {score}")

    average_score = sum(scores) / len(scores)
    best_score = max(scores)

    print()
    print(f"Average score over {episodes} games: {average_score:.2f}")
    print(f"Best score during evaluation: {best_score}")