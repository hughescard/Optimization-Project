# Optimization-Project

**Autor:** Guillermo Hughes Cardona — **Grupo:** C311  
**Repositorio:** https://github.com/hughescard/Optimization-Project

Proyecto de optimización (irrestricto) para minimizar
$$
f(x,y) = -\tan\Big(\tfrac{1}{(x+y)^2+1}\Big) - \tan\Big(\tfrac{1}{(x-y)^2+1}\Big).
$$

## Estructura
- `src/`: definición del modelo (`model.py`) y algoritmos (`methods.py`).
- `scripts/`: `run_experiments.py` genera figuras y tablas en `results/`.
- `notebooks/main.py`: script Jupytext-friendly para correr en Jupyter.
- `reporte/`: `reporte_tarea.tex` (LaTeX) + tabla *stub* que compila sin correr scripts.

## Requisitos
```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
