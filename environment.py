import random

from ple import PLE
from ple.games.snake import Snake

game = Snake()
p = PLE(game, fps=30, display_screen=True)

p.init()
reward = 0.0

for i in range(1000):
    if p.game_over():
        p.reset_game()

    observation = p.getScreenRGB()
    action = random.choice(p.getActionSet())
    reward = p.act(action)
