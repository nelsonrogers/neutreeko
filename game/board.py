import numpy as np
from copy import deepcopy
from .constants import (
    BOARD_SIZE, DIRECTIONS, DIR_NAMES, P1_PAWNS, P2_PAWNS,
    INITIAL_BOARD, NUM_ACTIONS
)


class Board:
    """Represents the Neutreeko game state."""

    def __init__(self, grid=None):
        if grid is None:
            self.grid = INITIAL_BOARD.copy()
        else:
            self.grid = np.array(grid, dtype=np.int8)

    def copy(self):
        return Board(self.grid.copy())

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def find_pawn(self, pawn_id):
        """Return (row, col) of pawn_id, or None if not on board."""
        pos = np.argwhere(self.grid == pawn_id)
        if len(pos) == 0:
            return None
        return tuple(pos[0])

    def pawns_of(self, player):
        """Return list of pawn IDs belonging to player (1 or 2)."""
        return list(P1_PAWNS) if player == 1 else list(P2_PAWNS)

    def pawn_positions(self, player):
        """Return sorted list of (row, col) for player's pawns."""
        return sorted(self.find_pawn(p) for p in self.pawns_of(player))

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def slide_destination(self, row, col, direction):
        """
        Slide from (row, col) in direction until hitting a wall or piece.
        Returns (new_row, new_col). Returns (row, col) if no movement possible.
        """
        dr, dc = DIRECTIONS[direction]
        r, c = row, col
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                break
            if self.grid[nr, nc] != 0:
                break
            r, c = nr, nc
        return r, c

    def get_legal_moves(self, player):
        """
        Return list of (pawn_id, direction, dest_row, dest_col) for all
        legal moves of the given player (1 or 2).
        """
        moves = []
        for pawn_id in self.pawns_of(player):
            pos = self.find_pawn(pawn_id)
            if pos is None:
                continue
            row, col = pos
            for direction in DIR_NAMES:
                dr, dc = DIRECTIONS[direction]
                # Check if movement is possible at all
                nr, nc = row + dr, col + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if self.grid[nr, nc] != 0:
                    continue
                dest_r, dest_c = self.slide_destination(row, col, direction)
                if (dest_r, dest_c) != (row, col):
                    moves.append((pawn_id, direction, dest_r, dest_c))
        return moves

    def apply_move(self, pawn_id, direction):
        """Return a new Board with the move applied."""
        new_board = self.copy()
        pos = new_board.find_pawn(pawn_id)
        if pos is None:
            return new_board
        row, col = pos
        dest_r, dest_c = new_board.slide_destination(row, col, direction)
        if (dest_r, dest_c) != (row, col):
            new_board.grid[row, col] = 0
            new_board.grid[dest_r, dest_c] = pawn_id
        return new_board

    # ------------------------------------------------------------------
    # Win detection
    # ------------------------------------------------------------------

    def _are_aligned(self, positions):
        """
        Check if 3 (row, col) positions are adjacent in a line
        (horizontal, vertical, or diagonal, spacing exactly 1).
        """
        positions = sorted(positions)
        (r0, c0), (r1, c1), (r2, c2) = positions
        # Direction from first to second
        dr = r1 - r0
        dc = c1 - c0
        if abs(dr) > 1 or abs(dc) > 1:
            return False
        if dr == 0 and dc == 0:
            return False
        # Third must continue the same direction
        return (r2 == r1 + dr) and (c2 == c1 + dc)

    def check_winner(self):
        """Return 1 if player 1 wins, 2 if player 2 wins, 0 otherwise."""
        for player in (1, 2):
            positions = [self.find_pawn(p) for p in self.pawns_of(player)]
            if any(p is None for p in positions):
                continue
            if self._are_aligned(positions):
                return player
        return 0

    # ------------------------------------------------------------------
    # DQN state encoding
    # ------------------------------------------------------------------

    def encode(self, perspective_player):
        """
        Encode board as a flat float32 array of length 75 (5×5×3).
        Channel 0: empty cells
        Channel 1: current player's pawns
        Channel 2: opponent's pawns
        perspective_player: which player's perspective (1 or 2)
        """
        state = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        own_pawns = set(self.pawns_of(perspective_player))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = self.grid[r, c]
                if v == 0:
                    state[r, c, 0] = 1.0
                elif v in own_pawns:
                    state[r, c, 1] = 1.0
                else:
                    state[r, c, 2] = 1.0
        return state.flatten()

    # ------------------------------------------------------------------
    # Action index helpers (for DQN)
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_move(action_idx, player):
        """
        Convert flat action index (0-23) to (pawn_id, direction).
        Actions: pawn_index * 8 + direction_index
        pawn_index 0,1,2 maps to player's pawns in order.
        """
        pawns = list(P1_PAWNS) if player == 1 else list(P2_PAWNS)
        pawn_idx = action_idx // 8
        dir_idx = action_idx % 8
        return pawns[pawn_idx], DIR_NAMES[dir_idx]

    def legal_action_mask(self, player):
        """Return boolean array of length 24: True where action is legal."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        legal = self.get_legal_moves(player)
        pawns = list(P1_PAWNS) if player == 1 else list(P2_PAWNS)
        for pawn_id, direction, _, _ in legal:
            pawn_idx = pawns.index(pawn_id)
            dir_idx = DIR_NAMES.index(direction)
            mask[pawn_idx * 8 + dir_idx] = True
        return mask

    def __repr__(self):
        return f"Board(\n{self.grid}\n)"
