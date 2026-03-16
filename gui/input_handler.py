import pygame
import puyotan_native as p

class GameplayController:
    """
    Translates hardware events (keys, buttons) into ViewModel commands.
    """
    def __init__(self, view_model):
        self.vm = view_model

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            # 1P Keys
            self._handle_player_key(0, event.key, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_z, pygame.K_DOWN)
            # 2P Keys (WASD)
            self._handle_player_key(1, event.key, pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_q, pygame.K_s)

    def _handle_player_key(self, pid, key, k_left, k_right, k_rot_r, k_rot_l, k_drop):
        if key == k_left:
            self.vm.move_player(pid, -1)
        elif key == k_right:
            self.vm.move_player(pid, 1)
        elif key == k_rot_r:
            self.vm.rotate_player(pid, 1)
        elif key == k_rot_l:
            self.vm.rotate_player(pid, -1)
        elif key == k_drop:
            self.vm.confirm_player(pid)

    def handle_button_click(self, button_id):
        if button_id == "restart":
            self.vm.restart()
        elif button_id.startswith("p1_"):
            self._handle_action(0, button_id[3:])
        elif button_id.startswith("p2_"):
            self._handle_action(1, button_id[3:])

    def _handle_action(self, pid, action_name):
        if action_name == "left":
            self.vm.move_player(pid, -1)
        elif action_name == "right":
            self.vm.move_player(pid, 1)
        elif action_name == "rot_r":
            self.vm.rotate_player(pid, 1)
        elif action_name == "rot_l":
            self.vm.rotate_player(pid, -1)
        elif action_name == "drop":
            self.vm.confirm_player(pid)
