import pygame
import puyotan_native as p
from . import config

class Button:
    def __init__(self, rect, text, button_id):
        self.rect = pygame.Rect(rect)
        self.id = button_id
        # Use symbol if available
        self.text_label = config.BUTTON_SYMBOLS.get(button_id.split("_")[-1], text)
        self.color = config.COLORS["Button"]
        self.hover_color = config.COLORS["ButtonHover"]
        self.text_color = config.COLORS["ButtonText"]
        self.font = pygame.font.SysFont("segoeuisymbol", 24) # Segoe UI Symbol has good arrow support on Windows
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, config.COLORS["Text"], self.rect, 2, border_radius=5)
        
        text_surf = self.font.render(self.text_label, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        return self.is_hovered

class PuyotanView:
    """
    Responsible for drawing the game state using the ViewModel.
    """
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font
        self.buttons = []
        self._init_buttons()

    def _init_buttons(self):
        # Restart Button
        self.buttons.append(Button(
            config.RESTART_BUTTON_RECT, "RESTART", "restart"
        ))
        
        # 1P Buttons
        for x, y, label, bid in config.P1_CONTROL_BUTTONS:
            self.buttons.append(Button(
                (x, y, config.BUTTON_WIDTH, config.BUTTON_HEIGHT), label, f"p1_{bid}"
            ))
            
        # 2P Buttons
        for x, y, label, bid in config.P2_CONTROL_BUTTONS:
            self.buttons.append(Button(
                (x, y, config.BUTTON_WIDTH, config.BUTTON_HEIGHT), label, f"p2_{bid}"
            ))

    def draw(self, vm):
        self.screen.fill(config.COLORS["Background"])
        
        # Draw Text / UI
        status_text = self.font.render(f"Status: {vm.status_text}", True, config.COLORS["Text"])
        frame_text = self.font.render(f"Turn: {vm.frame}", True, config.COLORS["Text"])
        self.screen.blit(status_text, (20, 15))
        self.screen.blit(frame_text, (20, 50))

        # Draw P1 Board
        self._draw_board(vm, 0, config.P1_OFFSET_X, config.P1_OFFSET_Y)
        
        # Draw P2 Board
        self._draw_board(vm, 1, config.P2_OFFSET_X, config.P2_OFFSET_Y)

        # Draw Active Piece / Confirmed Status
        for pid, offset in [(0, config.P1_OFFSET_X), (1, config.P2_OFFSET_X)]:
            state = vm.players[pid]
            
            if state.has_decision:
                # Draw "Ghost" active piece (pre-blended by ViewModel)
                self._draw_active_piece(
                    state.x, state.rotation, offset, config.P1_OFFSET_Y, 
                    state.ghost_color_axis, state.ghost_color_sub
                )
                
                if state.confirmed:
                    msg = self.font.render("READY", True, (100, 255, 100))
                    self.screen.blit(msg, (offset + 100, config.P1_OFFSET_Y - 45))

            # Draw Next Pieces
            self._draw_next_pieces(vm, pid, offset, config.P1_OFFSET_Y)

        # Draw Buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.check_hover(mouse_pos)
            btn.draw(self.screen)

    def _draw_board(self, vm, player_id, offset_x, offset_y):
        board = vm.get_player_field(player_id)
        score = vm.get_player_score(player_id)
        non_active_ojama, active_ojama = vm.get_player_ojama(player_id)
        state = vm.players[player_id]
        
        # Draw player label
        label = self.font.render(f"PLAYER {player_id + 1}", True, config.COLORS["Text"])
        self.screen.blit(label, (offset_x, offset_y - 45))

        # Draw Background Grid
        for y in range(config.VISIBLE_HEIGHT):
            for x in range(config.BOARD_WIDTH):
                rect = pygame.Rect(
                    offset_x + x * config.CELL_SIZE,
                    offset_y + (config.VISIBLE_HEIGHT - 1 - y) * config.CELL_SIZE,
                    config.CELL_SIZE,
                    config.CELL_SIZE
                )
                pygame.draw.rect(self.screen, config.COLORS["Grid"], rect, 1)

                cell = board.get(x, y)
                if cell != p.Cell.Empty:
                    color = config.COLOR_MAP.get(cell, (255, 255, 255))
                    center = (
                        offset_x + x * config.CELL_SIZE + config.CELL_SIZE // 2,
                        offset_y + (config.VISIBLE_HEIGHT - 1 - y) * config.CELL_SIZE + config.CELL_SIZE // 2
                    )
                    pygame.draw.circle(self.screen, color, center, config.CELL_SIZE // 2 - 2)

        # Draw Score and Ojama
        score_text = self.font.render(f"Score: {score}", True, config.COLORS["Text"])
        ojama_text = self.font.render(f"Ojama: {non_active_ojama} / {active_ojama}", True, config.COLORS["Text"])
        rigid_text = self.font.render(f"Rigid: {state.rigid_frames}", True, config.COLORS["Text"])
        
        self.screen.blit(score_text, (offset_x, offset_y + config.VISIBLE_HEIGHT * config.CELL_SIZE + 10))
        self.screen.blit(ojama_text, (offset_x, offset_y + config.VISIBLE_HEIGHT * config.CELL_SIZE + 35))
        self.screen.blit(rigid_text, (offset_x, offset_y + config.VISIBLE_HEIGHT * config.CELL_SIZE + 60))

    def _draw_next_pieces(self, vm, pid, offset_x, offset_y):
        # NEXT
        next_p = vm.get_next_piece(pid, 1)
        nx = offset_x + config.NEXT_OFFSET_X
        ny = offset_y + config.NEXT_OFFSET_Y
        self._draw_static_piece(next_p, nx, ny)
        
        # DOUBLE NEXT
        dnext_p = vm.get_next_piece(pid, 2)
        dnx = offset_x + config.DOUBLE_NEXT_OFFSET_X
        dny = offset_y + config.DOUBLE_NEXT_OFFSET_Y
        self._draw_static_piece(dnext_p, dnx, dny)

    def _draw_static_piece(self, piece, px, py):
        # Draw next tsumo vertically
        c1 = config.COLOR_MAP.get(piece.axis, (255, 255, 255))
        c2 = config.COLOR_MAP.get(piece.sub, (255, 255, 255))
        pygame.draw.circle(self.screen, c1, (px + config.CELL_SIZE//2, py + config.CELL_SIZE//2), config.CELL_SIZE//2 - 2)
        pygame.draw.circle(self.screen, c2, (px + config.CELL_SIZE//2, py - config.CELL_SIZE//2), config.CELL_SIZE//2 - 2)

    def _draw_active_piece(self, x, rot, offset_x, offset_y, color_axis, color_sub):
        # Move it slightly above row 12 (the first spawn row)
        axis_pixel_y = offset_y - config.CELL_SIZE // 2
        axis_pixel_x = offset_x + x * config.CELL_SIZE + config.CELL_SIZE // 2
        
        sub_rel_x = 0
        sub_rel_y = 0
        if rot == p.Rotation.Up:
            sub_rel_y = -config.CELL_SIZE
        elif rot == p.Rotation.Right:
            sub_rel_x = config.CELL_SIZE
        elif rot == p.Rotation.Down:
            sub_rel_y = config.CELL_SIZE
        elif rot == p.Rotation.Left:
            sub_rel_x = -config.CELL_SIZE

        # Draw axis
        pygame.draw.circle(self.screen, color_axis, (axis_pixel_x, axis_pixel_y), config.CELL_SIZE // 2 - 2, 2)
        # Draw sub
        pygame.draw.circle(self.screen, color_sub, (axis_pixel_x + sub_rel_x, axis_pixel_y + sub_rel_y), config.CELL_SIZE // 2 - 2, 2)
