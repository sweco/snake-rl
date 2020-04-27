import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical


class Agent(nn.Module):
    def __init__(self, in_dim, channels, actions):
        super(Agent, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actions = np.array(actions)

        self.c1 = nn.Conv2d(in_dim[2] if len(in_dim) >= 3 else 1, channels, 9, padding=4)
        self.p1 = nn.MaxPool2d(8)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(8 * 8 * channels, len(actions))

        self.to(self.device)

    def forward(self, x: Tensor):
        h = self.c1(x)  # (b, 1, 64, 64)
        h = self.p1(h)  # (b, ch, 64, 64)
        h = self.relu(h)  # (b, ch, 8, 8)
        h = h.view(h.shape[0], -1)
        l = self.l2(h)

        c = Categorical(logits=l)
        a = c.sample()
        p = c.log_prob(a)

        return self.actions[a], p
