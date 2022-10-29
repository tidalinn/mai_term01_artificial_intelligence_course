# plotting_functions module
import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, Tuple


def get_grid(radius: int, grid_step: float, f: Mapping) -> (np.array, np.array, np.array):
    """ Create grid of dots for plotting the 3D chart.

    Args:
        radius (int): limit of values
        grid_step (float): step to make between limits
        f (Mapping): executing function

    Returns:
        np.array: x values
        np.array: y values
        np.array: z values
    """

    try:
        samples = np.arange(-radius, radius, grid_step)
        x, y = np.meshgrid(samples, samples)

        return x, y, f(x, y)

    except:
        print('Убедитесь в корректности переданных аргументов')


def draw_chart(grid: np.array, point: Tuple = (None, None)):
    """ Draw chart.

    Args:
        grid (np.array): points coords
        point (Tuple): local minimum (Default = (None, None))

    Result:
        chart
    """

    try:
        grid_x, grid_y, grid_z = grid

        plt.rcParams.update({
            'figure.figsize': (4, 4),
            'figure.dpi': 200,
            'xtick.labelsize': 4,
            'ytick.labelsize': 4
        })

        ax = plt.figure().add_subplot(111, projection='3d')

        if point != (None, None):
            point_x, point_y, point_z = point
            ax.scatter(point_x, point_y, point_z, color='red')
            
        ax.plot_surface(grid_x, grid_y, grid_z, rstride=5, cstride=5, alpha=0.7)
        
        plt.show()
    
    except:
        print('Убедитесь в корректности переданных аргументов')