"""
generator_2d.py — Генерація двовимірної нормальної вибірки
методом лінійного перетворення двох незалежних стандартних нормальних величин
(розклад Холецького коваріаційної матриці).
"""

import numpy as np


def generate_2d_normal(mx: float, my: float, sx: float, sy: float, r: float, n: int, seed: int = None):
    """
    Генерує N пар (x, y) із параметрами (mx, my, sx, sy, r).
    
    Модель:
        Z1, Z2 ~ N(0, 1) незалежні
        X = mx + sx * Z1
        Y = my + sy * (r * Z1 + sqrt(1 - r^2) * Z2)
    """
    if seed is not None:
        np.random.seed(seed)

    if sx <= 0 or sy <= 0:
        raise ValueError("СКВ sx та sy повинні бути строго додатними!")
    if not (-1.0 < r < 1.0):
        raise ValueError("Коефіцієнт кореляції r повинен належати інтервалу (-1; 1)!")
    if n < 10:
        raise ValueError("Обсяг вибірки N повинен бути не меншим за 10!")

    z1 = np.random.normal(0.0, 1.0, size=n)
    z2 = np.random.normal(0.0, 1.0, size=n)

    x = mx + sx * z1
    y = my + sy * (r * z1 + np.sqrt(1.0 - r ** 2) * z2)

    return x, y