import numpy as np
import pandas as pd
import math
import random

from typing import Tuple, Mapping


def mccormick_loss(x, y):
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1


def mccormick_partials(x, y):
    partial_x = 2 * x - 2 * y + np.cos(x + y) - 1.5
    partial_y = -2 * x + 2 * y + np.cos(x + y) + 2.5
    
    return np.array([partial_x, partial_y])


def izom_loss(x, y):
    return -np.cos(x) * np.cos(x) * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))


def izom_partials(x, y):
    partial_x = -(-2 * x + 2 * np.pi) * np.exp((-(x - np.pi) ** 2 - (y - np.pi) ** 2)) * \
                np.cos(x) ** 2 + 2 * np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2) * \
                np.sin(x) * np.cos(x)
    
    partial_y = -(-2 * y + 2 * np.pi) * \
                np.exp(-(x - np.pi) ** 2 - (y - np.pi) ** 2) * \
                np.cos(x) ** 2
    
    return np.array([partial_x, partial_y])


def rastrigin_loss(*X, A: int = 10):
    return A + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in X])


def rastrigin_partials(x, y):
    A = 10
    partial_x = 2 * x + A * 2 * np.pi * np.sin(2 * np.pi * x)
    
    return np.array([partial_x, y])