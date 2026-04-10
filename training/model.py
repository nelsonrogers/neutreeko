"""
PyTorch DQN model definition.
Used only during training — not imported in the game itself.
"""
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Three-layer MLP mapping board state (75) → Q-values (24).
    Input:  5×5 board encoded as 3-channel one-hot = 75 floats
    Output: Q-value for each of 24 actions (3 pawns × 8 directions)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(75, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
        )

    def forward(self, x):
        return self.net(x)
