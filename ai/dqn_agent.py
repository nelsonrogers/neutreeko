"""
DQN inference agent — pure numpy, Pygbag/WASM compatible.
Loads pre-trained weights from assets/weights.npz.
"""
import os
import random
import numpy as np
from game.board import Board
from game.constants import NUM_ACTIONS


class DQNAgent:
    """
    Runs the forward pass of a trained DQN using only numpy.
    Network: Linear(75→128) → ReLU → Linear(128→64) → ReLU → Linear(64→24)
    """

    def __init__(self, weights_path=None):
        self.loaded = False
        if weights_path is None:
            # Try default location relative to project root
            here = os.path.dirname(os.path.abspath(__file__))
            weights_path = os.path.join(here, "..", "assets", "weights.npz")
        self._load(weights_path)

    def _load(self, path):
        if not os.path.exists(path):
            return
        data = np.load(path)
        try:
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.W3 = data["W3"].astype(np.float32)
            self.b3 = data["b3"].astype(np.float32)
            self.loaded = True
        except KeyError:
            pass

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    def forward(self, state):
        """state: float32 array of shape (75,). Returns Q-values (24,)."""
        x = self._relu(state @ self.W1.T + self.b1)
        x = self._relu(x @ self.W2.T + self.b2)
        return x @ self.W3.T + self.b3

    def get_move(self, board, player):
        """
        Return (pawn_id, direction) for the DQN's chosen move.
        Falls back to random legal move if weights not loaded.
        """
        moves = board.get_legal_moves(player)
        if not moves:
            return None

        if not self.loaded:
            pawn_id, direction, _, _ = random.choice(moves)
            return pawn_id, direction

        state = board.encode(player)
        q_values = self.forward(state)

        # Mask illegal actions with -inf
        mask = board.legal_action_mask(player)
        q_values[~mask] = -np.inf

        action = int(np.argmax(q_values))
        return Board.action_to_move(action, player)
