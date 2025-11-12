"""
Modelo del problema de optimización:
    f(x,y) = -tan(1/((x+y)^2 + 1)) - tan(1/((x-y)^2 + 1))

Incluye: f, gradiente y hessiana; utilidades g'(t) y g''(t).

Consultar procedimiento en `reports/reporte_tarea.tex`
"""

import math
import numpy as np

def gprime(t: float) -> float:
    """g'(t) para g(t) = -tan( 1/(t^2+1) )."""
    s = 1.0 / (t*t + 1.0)
    # derivada de -tan(s(t)) = -(sec^2(s) * s'(t))
    # s'(t) = -2t/(t^2+1)^2  => g'(t) = 2t / ((t^2+1)^2 * cos(s)^2)
    return (2.0 * t) / (((t*t) + 1.0)**2 * (math.cos(s)**2))

def gsecond(t: float) -> float:
    """g''(t) (expresión cerrada, ya simplificada)."""
    s = 1.0 / (t*t + 1.0)
    num = -6.0*(t**4) - 4.0*(t**2) - 8.0*(t**2)*math.tan(s) + 2.0
    den = (((t*t) + 1.0)**4) * (math.cos(s)**2)
    return num / den

def f_xy(x: float, y: float) -> float:
    """Función objetivo f(x,y)."""
    u = x + y
    v = x - y
    return -math.tan(1.0 / (u*u + 1.0)) - math.tan(1.0 / (v*v + 1.0))

def grad_xy(x: float, y: float) -> np.ndarray:
    """Gradiente de f(x,y)."""
    u = x + y
    v = x - y
    gp_u = gprime(u)
    gp_v = gprime(v)
    # [df/dx, df/dy] = [g'(u)+g'(v), g'(u)-g'(v)]
    return np.array([gp_u + gp_v, gp_u - gp_v], dtype=float)

def hess_xy(x: float, y: float) -> np.ndarray:
    """Hessiana de f(x,y) (2x2)."""
    u = x + y
    v = x - y
    g2u = gsecond(u)
    g2v = gsecond(v)
    # H = g''(u) * [[1,1],[1,1]] + g''(v) * [[1,-1],[-1,1]]
    H = np.array([[g2u + g2v, g2u - g2v],
                  [g2u - g2v, g2u + g2v]], dtype=float)
    return H
