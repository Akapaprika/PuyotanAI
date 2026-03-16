import pygame
import sys
from . import config
from .model import GameModel
from .view_model import PuyotanViewModel
from .input_handler import GameplayController
from .view import PuyotanView
import puyotan_native as p

def main():
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("Puyotan AI GUI (MVVM)")
    
    # Enable font system
    pygame.font.init()
    font = pygame.font.SysFont(None, 24)

    clock = pygame.time.Clock()
    
    # Initialize MVVM
    model = GameModel(seed=42)
    view_model = PuyotanViewModel(model)
    controller = GameplayController(view_model)
    view = PuyotanView(screen, font)

    running = True

    while running:
        current_time = pygame.time.get_ticks()
        
        # 1. Handle Events (Controller)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    for btn in view.buttons:
                        if btn.rect.collidepoint(event.pos):
                            controller.handle_button_click(btn.id)
            else:
                controller.handle_event(event)

        # 2. Update Logic (ViewModel)
        view_model.update(current_time)

        # 3. Render (View)
        view.draw(view_model)
        pygame.display.flip()

        # 4. Tick
        clock.tick(config.FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
