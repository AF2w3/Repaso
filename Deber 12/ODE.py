
from typing import Callable


def ODE_euler(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
    
    h = (b - a) / N
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        y += h * f(t, y)
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h



from math import factorial


def ODE_euler_nth(
    *,
    a: float,
    b: float,
    f: Callable[[float, float], float],
    f_derivatives: list[Callable[[float, float], float]],
    y_t0: float,
    N: int,
) -> tuple[list[float], list[float], float]:
   
    h = (b - a) / N
    t = a
    ts = [t]
    ys = [y_t0]

    for _ in range(N):
        y = ys[-1]
        T = f(t, y)
        ders = [
            h / factorial(m + 2) * mth_derivative(t, y)
            for m, mth_derivative in enumerate(f_derivatives)
        ]
        T += sum(ders)
        y += h * T
        ys.append(y)

        t += h
        ts.append(t)
    return ys, ts, h


import numpy as np

def interpolacion_lineal(y_1, y_2, x_val):
 
    # Convertir listas a arrays numpy
    y_1 = np.array(y_1)
    y_2 = np.array(y_2)

    # Usar la función interp1d de scipy para la interpolación lineal
    from scipy.interpolate import interp1d

    # Crear la función de interpolación
    f = interp1d(y_1, y_2, kind='linear', fill_value='extrapolate')

    # Obtener el valor interpolado
    y_val = f(x_val)

    return y_val







