import numpy as np
import pygame as pg
import math

# import numba
# import taichi as ti

# settings
res = width, height = 160, 100

class Fractal:
    def __init__(self, app) -> None:
        self.app = app
        # rgb
        self.screen_array = np.full((width, height, 3), [0, 0, 0], dtype=np.uint8)
    
    def render(self): # what to draw
        pass

    def update(self):
        self.render()
    
    def draw(self): # how to draw
        # send to buffer
        pg.surfarray.blit_array(self.app.screen, self.screen_array)
    
    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self) -> None:
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)
    
    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps()}')

# точка входа
if __name__ == '__main__':
    app = App()
    app.run()