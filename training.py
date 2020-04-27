import logging
import random

from ple import PLE
from ple.games.snake import Snake

logging.basicConfig(level=logging.INFO)


def play_episode(env, max_steps=1000):
    env.reset_game()

    step = 0
    for step in range(max_steps):
        if env.game_over():
            break

        observation = env.getScreenRGB()
        action = random.choice(env.getActionSet())
        reward = env.act(action)

    logging.info("Finished episode with %d steps", step)


if __name__ == '__main__':
    env = PLE(Snake(), fps=30, display_screen=True)
    env.init()

    play_episode(env)
