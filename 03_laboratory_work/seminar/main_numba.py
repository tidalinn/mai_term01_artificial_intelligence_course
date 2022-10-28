import numpy as np
import pygame as pg
import math
import numba

# import taichi as ti

# settings
res = width, height = 800, 450
# в отличие от компьютерной графики у фрактала другая система координат
offset = np.array([1.3 * width, height]) // 2
zoom = 2.2 / height
max_iter = 30

# texture
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1 # make it squared
texture_array = pg.surfarray.array3d(texture)

class Fractal:
    def __init__(self, app) -> None:
        self.app = app
        # rgb
        self.screen_array = np.full((width, height, 3), [0, 0, 0], dtype=np.uint8)
        self.x = np.linspace(0, width, num=width, dtype=np.float32)
        self.y = np.linspace(0, height, num=height, dtype=np.float32)
    
    @staticmethod
    @numba.njit(fastmath=True)
    def render(screen_array): # what to draw
        # голый питон
        for x in range(width):
            for y in range(height):
                # каждому пикселю экрана своё комплексное число
                c = (x - offset[0]) * zoom + 1j * (y - offset[1]) * zoom
                z = 0
                num_iter = 0

                for i in range(max_iter):
                    z = z ** 2 + c

                    # окружность радиуса 2 (из Мандельброта)
                    if z.real ** 2 + z.imag ** 2 > 4:
                        break
                    num_iter += 1
                
                col = int(texture_size * num_iter / max_iter)
                screen_array[x, y] = texture_array[col, col]

        return screen_array

    def update(self):
        self.screen_array = self.render(self.screen_array)
    
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