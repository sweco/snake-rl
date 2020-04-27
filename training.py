import logging

import numpy as np
import torch
from ple import PLE
from ple.games.snake import Snake

from agent import Agent

logging.basicConfig(level=logging.INFO)


def play_episode(env, agent, max_steps=1000):
    env.reset_game()

    step = 0
    log_probas, rewards = [], []
    for step in range(max_steps):
        if env.game_over():
            break

        observation = env.getScreenGrayscale()
        observation = observation.astype(np.float32) / 255
        observation = torch.tensor(observation, device=agent.device).view(1, 1, 64, 64)
        action, log_proba = agent.forward(observation)
        reward = env.act(action)

        log_probas.append(log_proba.view(1))
        rewards.append(reward)

    logging.info("Finished episode with %d steps and %d cumulative reward", step, sum(rewards))

    return torch.cat(log_probas), rewards


def discounted_rewards(rewards, gamma=0.99):
    res = []
    for r in reversed(rewards):
        cum_reward = res[0] if res else 0
        res.insert(0, gamma * cum_reward + r)

    return res


def train(env, agent):
    optimizer = torch.optim.Adam(agent.parameters())

    while True:
        agent.zero_grad()
        p, r = play_episode(env, agent)
        r = torch.tensor(discounted_rewards(r), device=agent.device)
        loss = - r * p
        loss = loss.mean()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    env = PLE(Snake(), fps=30, display_screen=True)
    env.init()
    agent = Agent(env.getScreenDims(), 16, env.getActionSet())

    train(env, agent)
