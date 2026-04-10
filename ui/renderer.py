"""
Pygame board renderer.
All drawing functions take a pygame Surface and game state.
"""
import pygame
from game.constants import (
    BOARD_SIZE, CELL_SIZE, BOARD_OFFSET_X, BOARD_OFFSET_Y,
    BG_COLOR, CELL_DARK, CELL_LIGHT, GRID_COLOR,
    P1_COLOR, P2_COLOR, SELECT_GLOW, TEXT_COLOR,
)


def cell_rect(row, col):
    """Return the pygame.Rect for grid cell (row, col)."""
    x = BOARD_OFFSET_X + col * CELL_SIZE
    y = BOARD_OFFSET_Y + row * CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)


def cell_center(row, col):
    r = cell_rect(row, col)
    return r.centerx, r.centery


def draw_board(surface, board, selected_pawn=None, legal_moves=None, anim=None):
    """
    Draw the full board:
      board         — Board instance
      selected_pawn — pawn_id currently selected (or None)
      legal_moves   — list of (pawn_id, direction, dest_r, dest_c) for selected
      anim          — dict {pawn_id, from_pos, to_pos, progress} for sliding anim
    """
    surface.fill(BG_COLOR)
    _draw_cells(surface)
    _draw_hints(surface, legal_moves)
    _draw_pieces(surface, board, selected_pawn, anim)
    _draw_grid(surface)


def _draw_cells(surface):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = CELL_LIGHT if (row + col) % 2 == 0 else CELL_DARK
            pygame.draw.rect(surface, color, cell_rect(row, col))


def _draw_grid(surface):
    for i in range(BOARD_SIZE + 1):
        x = BOARD_OFFSET_X + i * CELL_SIZE
        y = BOARD_OFFSET_Y + i * CELL_SIZE
        pygame.draw.line(surface, GRID_COLOR,
                         (x, BOARD_OFFSET_Y),
                         (x, BOARD_OFFSET_Y + BOARD_SIZE * CELL_SIZE), 2)
        pygame.draw.line(surface, GRID_COLOR,
                         (BOARD_OFFSET_X, y),
                         (BOARD_OFFSET_X + BOARD_SIZE * CELL_SIZE, y), 2)


def _draw_hints(surface, legal_moves):
    if not legal_moves:
        return
    hint_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    hint_surf.fill((0, 0, 0, 0))
    radius = CELL_SIZE // 6
    pygame.draw.circle(hint_surf, (255, 230, 80, 120),
                       (CELL_SIZE // 2, CELL_SIZE // 2), radius)

    seen = set()
    for _, _, dr, dc in legal_moves:
        if (dr, dc) not in seen:
            seen.add((dr, dc))
            r = cell_rect(dr, dc)
            surface.blit(hint_surf, (r.x, r.y))


def _draw_pieces(surface, board, selected_pawn, anim):
    from game.constants import P1_PAWNS, P2_PAWNS
    anim_pawn = anim["pawn_id"] if anim else None

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pawn_id = int(board.grid[row, col])
            if pawn_id == 0:
                continue
            if pawn_id == anim_pawn:
                continue  # drawn separately during animation

            color = P1_COLOR if pawn_id in P1_PAWNS else P2_COLOR
            selected = (pawn_id == selected_pawn)
            _draw_one_piece(surface, row, col, color, selected)

    # Draw animating piece at interpolated position
    if anim:
        fr, fc = anim["from_pos"]
        tr, tc = anim["to_pos"]
        t = anim["progress"]
        ir = fr + (tr - fr) * t
        ic = fc + (tc - fc) * t
        x = BOARD_OFFSET_X + ic * CELL_SIZE + CELL_SIZE // 2
        y = BOARD_OFFSET_Y + ir * CELL_SIZE + CELL_SIZE // 2
        pawn_id = anim["pawn_id"]
        color = P1_COLOR if pawn_id in P1_PAWNS else P2_COLOR
        _draw_piece_at_pixel(surface, x, y, color, selected=False)


def _draw_one_piece(surface, row, col, color, selected):
    cx, cy = cell_center(row, col)
    _draw_piece_at_pixel(surface, cx, cy, color, selected)


def _draw_piece_at_pixel(surface, cx, cy, color, selected):
    radius = CELL_SIZE // 2 - 8

    if selected:
        # Glow ring
        for offset in range(6, 0, -2):
            glow = (*SELECT_GLOW, max(0, 40 * offset))
            glow_surf = pygame.Surface((radius * 2 + 20, radius * 2 + 20), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow,
                               (radius + 10, radius + 10), radius + offset)
            surface.blit(glow_surf, (cx - radius - 10, cy - radius - 10))

    # Shadow
    shadow_surf = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(shadow_surf, (0, 0, 0, 80),
                       (radius + 3, radius + 6), radius)
    surface.blit(shadow_surf, (cx - radius - 3 + 2, cy - radius - 3 + 2))

    # Main circle
    pygame.draw.circle(surface, color, (cx, cy), radius)

    # Highlight
    highlight = tuple(min(255, c + 60) for c in color)
    pygame.draw.circle(surface, highlight,
                       (cx - radius // 4, cy - radius // 4), radius // 3)

    # Border
    border = tuple(max(0, c - 40) for c in color)
    pygame.draw.circle(surface, border, (cx, cy), radius, 2)


def draw_status_bar(surface, font, message, player_color=None):
    """Draw a status bar at the top of the window."""
    bar_rect = pygame.Rect(0, 0, surface.get_width(), BOARD_OFFSET_Y - 10)
    pygame.draw.rect(surface, BG_COLOR, bar_rect)

    if player_color:
        indicator_rect = pygame.Rect(14, 20, 16, 16)
        pygame.draw.circle(surface, player_color,
                           indicator_rect.center, 10)
        pygame.draw.circle(surface, (180, 180, 220),
                           indicator_rect.center, 10, 2)
        text_surf = font.render(message, True, TEXT_COLOR)
        surface.blit(text_surf, (40, 16))
    else:
        text_surf = font.render(message, True, TEXT_COLOR)
        surface.blit(text_surf, (14, 16))
