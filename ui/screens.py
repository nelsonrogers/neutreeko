"""
Menu and win/lose overlay screens.
"""
import pygame
from game.constants import (
    WINDOW_W, WINDOW_H, BG_COLOR, TEXT_COLOR,
    BTN_NORMAL, BTN_HOVER, BTN_TEXT,
    P1_COLOR, P2_COLOR, CELL_DARK,
)


class Button:
    def __init__(self, rect, label, description=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.description = description
        self.hovered = False

    def draw(self, surface, font, small_font):
        color = BTN_HOVER if self.hovered else BTN_NORMAL
        # Rounded rect
        pygame.draw.rect(surface, color, self.rect, border_radius=12)
        pygame.draw.rect(surface, (100, 130, 200), self.rect, 2, border_radius=12)

        label_surf = font.render(self.label, True, BTN_TEXT)
        lx = self.rect.centerx - label_surf.get_width() // 2
        ly = self.rect.centery - label_surf.get_height() // 2
        if self.description:
            ly -= 10
        surface.blit(label_surf, (lx, ly))

        if self.description:
            desc_surf = small_font.render(self.description, True, (150, 170, 210))
            dx = self.rect.centerx - desc_surf.get_width() // 2
            surface.blit(desc_surf, (dx, ly + label_surf.get_height() + 4))

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def is_clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


def build_menu_buttons():
    cx = WINDOW_W // 2
    buttons = [
        Button((cx - 130, 280, 260, 60), "Easy",   "Random moves"),
        Button((cx - 130, 360, 260, 60), "Medium", "Alpha-beta search"),
        Button((cx - 130, 440, 260, 60), "Hard",   "Deep Q-Network AI"),
    ]
    return buttons


def draw_menu(surface, font_large, font_medium, font_small, buttons, mouse_pos):
    surface.fill(BG_COLOR)

    # Title
    title = font_large.render("NEUTREEKO", True, (200, 220, 255))
    surface.blit(title, (WINDOW_W // 2 - title.get_width() // 2, 80))

    subtitle = font_small.render("Align your 3 pieces to win", True, (130, 150, 200))
    surface.blit(subtitle, (WINDOW_W // 2 - subtitle.get_width() // 2, 140))

    # Pawn previews
    _draw_pawn_preview(surface, P1_COLOR, WINDOW_W // 2 - 50, 200)
    _draw_pawn_preview(surface, P2_COLOR, WINDOW_W // 2 + 10, 200)

    # Difficulty label
    diff_label = font_medium.render("Choose difficulty:", True, TEXT_COLOR)
    surface.blit(diff_label, (WINDOW_W // 2 - diff_label.get_width() // 2, 240))

    for btn in buttons:
        btn.update(mouse_pos)
        btn.draw(surface, font_medium, font_small)

    # Instructions at bottom
    lines = [
        "Click a piece to select  ·  Click a destination to move",
        "Slide until a wall or another piece stops you",
    ]
    y = 530
    for line in lines:
        s = font_small.render(line, True, (100, 120, 170))
        surface.blit(s, (WINDOW_W // 2 - s.get_width() // 2, y))
        y += 22


def _draw_pawn_preview(surface, color, x, y):
    pygame.draw.circle(surface, color, (x, y), 16)
    highlight = tuple(min(255, c + 60) for c in color)
    pygame.draw.circle(surface, highlight, (x - 5, y - 5), 5)
    border = tuple(max(0, c - 40) for c in color)
    pygame.draw.circle(surface, border, (x, y), 16, 2)


def draw_win_screen(surface, font_large, font_medium, font_small,
                    winner_label, winner_color, play_again_btn, menu_btn, mouse_pos):
    # Semi-transparent overlay
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((10, 10, 30, 200))
    surface.blit(overlay, (0, 0))

    # Winner text
    win_surf = font_large.render("WINNER", True, winner_color)
    surface.blit(win_surf, (WINDOW_W // 2 - win_surf.get_width() // 2, 160))

    name_surf = font_large.render(winner_label, True, (255, 255, 255))
    surface.blit(name_surf, (WINDOW_W // 2 - name_surf.get_width() // 2, 230))

    # Large pawn graphic
    _draw_pawn_preview(surface, winner_color, WINDOW_W // 2, 320)
    pygame.draw.circle(surface, winner_color, (WINDOW_W // 2, 320), 32)
    highlight = tuple(min(255, c + 60) for c in winner_color)
    pygame.draw.circle(surface, highlight, (WINDOW_W // 2 - 10, 310), 10)

    play_again_btn.update(mouse_pos)
    play_again_btn.draw(surface, font_medium, font_small)
    menu_btn.update(mouse_pos)
    menu_btn.draw(surface, font_medium, font_small)


def build_postgame_buttons():
    cx = WINDOW_W // 2
    play_again = Button((cx - 130, 390, 260, 55), "Play Again")
    menu       = Button((cx - 130, 460, 260, 55), "Main Menu")
    return play_again, menu
