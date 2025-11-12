#!/usr/bin/env python3
# Genera figuras, tablas y >=400 experimentos (principales + aleatorios)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- bootstrap de ruta para importar src/ si ejecutas como script
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import f_xy, grad_xy
from src.methods import gradient_descent, newton_regularizado

DIR_FIG = ROOT / "results" / "figuras"
DIR_TAB = ROOT / "results" / "tablas"
DIR_FIG.mkdir(parents=True, exist_ok=True)
DIR_TAB.mkdir(parents=True, exist_ok=True)

# ---------- utilidades ----------
def grid_values(xmin=-4, xmax=4, ymin=-4, ymax=4, n=250):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    XX, YY = np.meshgrid(xs, ys)
    fvec = np.vectorize(f_xy)
    ZZ = fvec(XX, YY)
    return XX, YY, ZZ

def plot_contours_with_path(path, title, outfile):
    XX, YY, ZZ = grid_values()
    plt.figure()
    cs = plt.contour(XX, YY, ZZ, levels=30)
    plt.clabel(cs, inline=True, fontsize=8)
    P = np.asarray(path)
    plt.plot(P[:,0], P[:,1], marker='o', linewidth=1.5)
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(DIR_FIG / outfile, dpi=150)
    plt.close()

def plot_surface(outfile):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    XX, YY, ZZ = grid_values()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=True)
    ax.set_title("Superficie de f(x,y)"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
    plt.tight_layout()
    fig.savefig(DIR_FIG / outfile, dpi=150); plt.close(fig)

def record_result(method_name, x0, y0, rdict, tag):
    gx_fin = grad_xy(rdict["x"], rdict["y"])
    return {
        "tag": tag,                # principal_axes / principal_diagonals / alpha_sweep / random
        "method": method_name,     # nombre método
        "x0": x0, "y0": y0,
        "xf": rdict["x"], "yf": rdict["y"],
        "f(xf)": rdict["f"],
        "iters": rdict["iters"],
        "grad_norm_fin": float(np.linalg.norm(gx_fin)),
    }

# ---------- principales: diseños de inicios ----------
def principal_starts_axes(radii=(1,2,5,10,20)):
    S = []
    for r in radii:
        S += [( r,0),( -r,0),(0, r),(0,-r)]
    return S  # 5*4 = 20 puntos

def principal_starts_diagonals(radii=(1,2,5,10,20)):
    S = []
    for r in radii:
        S += [( r, r),( r,-r),(-r, r),(-r,-r)]
    return S  # 5*4 = 20 puntos

def alpha_sweep_starts():
    return [(5.0,-5.0),(10.0,0.0)]

ALPHAS = [0.05, 0.1, 0.2, 0.5]

# ---------- corrida masiva ----------
def run_mass_experiments(N_random_newton=250, N_random_gd=50, seed=42):
    rows = []

    # 0) Figuras de apoyo
    plot_surface("fig_superficie.png")
    gd_arm = gradient_descent(2,2, step="armijo")
    plot_contours_with_path(gd_arm["path"], "Gradiente con Armijo (inicio (2,2))", "fig_contorno_gd_armijo.png")
    nt_reg = newton_regularizado(2,2)
    plot_contours_with_path(nt_reg["path"], "Newton regularizado (inicio (2,2))", "fig_contorno_newton.png")
    gd_fix = gradient_descent(5,-5, step="constant", alpha=0.1)
    plot_contours_with_path(gd_fix["path"], "Gradiente paso fijo α=0.1 (inicio (5,-5))", "fig_contorno_gd_fijo.png")

    # 1) PRINCIPAL — Ejes
    for (x0,y0) in principal_starts_axes():
        r = gradient_descent(x0,y0, step="armijo")
        rows.append(record_result(r["method"], x0,y0, r, tag="principal_axes"))
        r = gradient_descent(x0,y0, step="constant", alpha=0.1)
        rows.append(record_result(r["method"], x0,y0, r, tag="principal_axes"))
        r = newton_regularizado(x0,y0)
        rows.append(record_result(r["method"], x0,y0, r, tag="principal_axes"))

    # 2) PRINCIPAL — Diagonales
    for (x0,y0) in principal_starts_diagonals():
        r = gradient_descent(x0,y0, step="armijo")
        rows.append(record_result(r["method"], x0,y0, r, tag="principal_diagonals"))
        r = gradient_descent(x0,y0, step="constant", alpha=0.1)
        rows.append(record_result(r["method"], x0,y0, r, tag="principal_diagonals"))
        r = newton_regularizado(x0,y0)
        rows.append(record_result(r["method"], x0,y0, r, tag="principal_diagonals"))

    # 3) PRINCIPAL — Barrido de alfas (gradiente paso fijo)
    for (x0,y0) in alpha_sweep_starts():
        for a in ALPHAS:
            r = gradient_descent(x0,y0, step="constant", alpha=a)
            rows.append(record_result(f"Gradiente (α={a})", x0,y0, r, tag="alpha_sweep"))
        # comparación contra Armijo y Newton
        r = gradient_descent(x0,y0, step="armijo")
        rows.append(record_result(r["method"], x0,y0, r, tag="alpha_sweep"))
        r = newton_regularizado(x0,y0)
        rows.append(record_result(r["method"], x0,y0, r, tag="alpha_sweep"))

    # 4) ALEATORIOS — Newton (robusto y rápido)
    rng = np.random.default_rng(seed)
    starts = rng.uniform(-100.0, 100.0, size=(N_random_newton,2))
    for x0,y0 in starts:
        r = newton_regularizado(float(x0), float(y0))
        rows.append(record_result(r["method"], float(x0), float(y0), r, tag="random_newton"))

    # 5) ALEATORIOS — Gradiente+Armijo (unas cuantas para mostrar lentitud)
    starts = rng.uniform(-100.0, 100.0, size=(N_random_gd,2))
    for x0,y0 in starts:
        r = gradient_descent(float(x0), float(y0), step="armijo")
        rows.append(record_result(r["method"], float(x0), float(y0), r, tag="random_gd"))

    df = pd.DataFrame(rows)
    df.to_csv(DIR_TAB / "experiments_summary.csv", index=False)
    return df

def small_table_preview(df):
    # pequeña tabla resumida por algunos inicios fijos (como antes)
    keep = []
    fixed_starts = { "(2,2)":(2.0,2.0), "(-2,3)":(-2.0,3.0), "(5,-5)":(5.0,-5.0), "(0.5,-1)":(0.5,-1.0), "(10,0)":(10.0,0.0) }
    for label, (x0,y0) in fixed_starts.items():
        block = df[(np.isclose(df["x0"],x0)) & (np.isclose(df["y0"],y0))]
        # si hay múltiples (por distintas etiquetas), toma los 3 métodos principales del primero que aparezca
        sample = []
        for m in ["Gradiente (Armijo)","Gradiente (α=0.1)","Newton regularizado (Armijo)","Newton (regularizado, Armijo)","Newton regularizado (Armijo)"]:
            if (block["method"]==m).any():
                sample.append(block[block["method"]==m].iloc[0])
        keep += sample
    if keep:
        df_small = pd.DataFrame(keep)
        # columnas amigables
        out = df_small[["x0","y0","method","xf","yf","f(xf)","iters"]].copy()
        out = out.rename(columns={"x0":"Inicio x0","y0":"Inicio y0","method":"Método","xf":"x*","yf":"y*","f(xf)":"f(x*)","iters":"Iteraciones"})
        out.to_csv(DIR_TAB / "tabla_resultados_peq.csv", index=False)
        with open(DIR_TAB / "tabla_resultados_peq.tex", "w", encoding="utf-8") as fh:
            fh.write(out.to_latex(index=False, float_format="%.6g"))
        print("Tabla pequeña guardada:", DIR_TAB / "tabla_resultados_peq.tex")

def main():
    df = run_mass_experiments(
        N_random_newton=250,  # puedes subir/bajar
        N_random_gd=50,       # unas cuantas para contraste
        seed=42
    )
    # reporte de conteos
    print("Total de experimentos:", len(df))
    print(df["tag"].value_counts())
    print(df["method"].value_counts())

    # tabla chica como antes (para el PDF)
    small_table_preview(df)

if __name__ == "__main__":
    main()
