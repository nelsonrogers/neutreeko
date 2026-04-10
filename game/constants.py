# Board dimensions
BOARD_SIZE = 5

# Direction vectors: (row_delta, col_delta)
DIRECTIONS = {
    "R":  (0,  1),
    "L":  (0, -1),
    "U":  (-1, 0),
    "D":  (1,  0),
    "UR": (-1, 1),
    "UL": (-1,-1),
    "DR": (1,  1),
    "DL": (1, -1),
}
DIR_NAMES = list(DIRECTIONS.keys())  # ordered list for action indexing

# Player identifiers
P1 = 1  # AI / opponent
P2 = 2  # human

# Pawn IDs: player 1 owns pawns 1,2,3  player 2 owns pawns 4,5,6
P1_PAWNS = (1, 2, 3)
P2_PAWNS = (4, 5, 6)

# Action space: 3 pawns × 8 directions = 24
NUM_ACTIONS = 24

# Initial board layout
import numpy as np
INITIAL_BOARD = np.array([
    [0, 1, 0, 2, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 5, 0, 6, 0],
], dtype=np.int8)

# --- UI constants ---
WINDOW_W = 520
WINDOW_H = 620
BOARD_OFFSET_X = 10
BOARD_OFFSET_Y = 80
CELL_SIZE = 100

# Colors
BG_COLOR        = (26,  26,  46)   # #1a1a2e  dark navy
CELL_DARK       = (22,  33,  62)   # #16213e
CELL_LIGHT      = (15,  52,  96)   # #0f3460
GRID_COLOR      = (80,  80, 120)
P1_COLOR        = (233, 69,  96)   # #e94560  coral-red
P2_COLOR        = (79, 195, 247)   # #4fc3f7  cyan
HINT_COLOR      = (255, 230,  80, 100)   # yellow, semi-transparent
SELECT_GLOW     = (255, 220,  50)
TEXT_COLOR      = (220, 220, 240)
OVERLAY_COLOR   = (10,  10,  30, 200)
BTN_NORMAL      = (40,  60, 100)
BTN_HOVER       = (60,  90, 150)
BTN_TEXT        = (220, 220, 240)
