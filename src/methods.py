"""
Implementación de algoritmos:
  - Descenso por gradiente (paso fijo y backtracking Armijo)
  - Newton regularizado + Armijo (tipo Levenberg–Marquardt)
Devuelven dict con: x, y, f, iters, fevals, path (np.ndarray [k,3])
"""

import math
import numpy as np
from .model import f_xy, grad_xy, hess_xy

def backtracking_armijo(x, y, d, fx, gx, c=1e-4, tau=0.5, t0=1.0):
    """Búsqueda de paso t que satisface la condición de Armijo."""
    t = t0
    gd = float(gx @ d)
    while True:
        xn = x + t*d[0]
        yn = y + t*d[1]
        fn = f_xy(xn, yn)
        if fn <= fx + c*t*gd:
            return t, fn, xn, yn
        t *= tau
        if t < 1e-12:  # salvaguarda numérica
            return t, fn, xn, yn

def gradient_descent(x0, y0, step="armijo", alpha=0.1, tol=1e-10, maxit=10_000):
    """Descenso por gradiente (Armijo o paso fijo)."""
    x, y = float(x0), float(y0)
    path = [(x, y, f_xy(x, y))]
    it = 0
    fevals = 1
    while it < maxit:
        fx = f_xy(x, y); fevals += 1
        g = grad_xy(x, y)
        if np.linalg.norm(g) < tol:
            break
        d = -g
        if step.lower() == "armijo":
            t, fn, xn, yn = backtracking_armijo(x, y, d, fx, g, c=1e-4, tau=0.5, t0=1.0)
            # (conteo aproximado de evals por backtracking)
            fevals += 1
        else:  # paso fijo
            t = float(alpha)
            xn = x + t*d[0]; yn = y + t*d[1]
            fn = f_xy(xn, yn); fevals += 1
        x, y = xn, yn
        path.append((x, y, fn))
        it += 1
        if np.linalg.norm([t*d[0], t*d[1]]) < tol:
            break
    return {
        "method": f"Gradiente ({'Armijo' if step.lower()=='armijo' else f'α={alpha:g}'})",
        "x": x, "y": y, "f": f_xy(x, y),
        "iters": it, "fevals": fevals,
        "path": np.array(path, dtype=float)
    }

def newton_regularizado(x0, y0, tol=1e-10, maxit=200):
    """Newton con regularización (LM) + Armijo."""
    x, y = float(x0), float(y0)
    path = [(x, y, f_xy(x, y))]
    it = 0
    fevals = 1
    while it < maxit:
        fx = f_xy(x, y); fevals += 1
        g = grad_xy(x, y)
        if np.linalg.norm(g) < tol:
            break
        H = hess_xy(x, y)
        # regularización para SPD: mu = max(0, -lambda_min) + eps
        evals = np.linalg.eigvals(H)
        lam_min = float(np.min(np.real(evals)))
        mu = 0.0
        if lam_min <= 1e-12:
            mu = (-lam_min) + 1e-6
        Hsp = H + mu*np.eye(2)
        try:
            d = -np.linalg.solve(Hsp, g)
        except np.linalg.LinAlgError:
            d = -g
        if float(g @ d) >= 0:
            d = -g
        # Armijo
        t, fn, xn, yn = backtracking_armijo(x, y, d, fx, g, c=1e-4, tau=0.5, t0=1.0)
        fevals += 1
        x, y = xn, yn
        path.append((x, y, fn))
        it += 1
        if np.linalg.norm([t*d[0], t*d[1]]) < tol:
            break
    return {
        "method": "Newton (regularizado, Armijo)",
        "x": x, "y": y, "f": f_xy(x, y),
        "iters": it, "fevals": fevals,
        "path": np.array(path, dtype=float)
    }
