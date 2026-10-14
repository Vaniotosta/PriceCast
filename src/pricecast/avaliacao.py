"""Métricas e comparação por grupo (produtos conhecidos x inéditos)."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metricas(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "registros": int(len(y)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "erro_pct_mediano": float(np.median(np.abs(y - p) / y) * 100),
    }


def comparar_por_grupo(y, previsoes: dict, conhecido) -> pd.DataFrame:
    """previsoes = {'nome do modelo': array}. Devolve uma tabela por grupo."""
    y, conhecido = np.asarray(y), np.asarray(conhecido, bool)
    linhas = []
    for grupo, mascara in [("Todos", np.ones(len(y), bool)),
                           ("Conhecidos", conhecido), ("Inéditos", ~conhecido)]:
        if mascara.sum() < 2:
            continue
        for nome, p in previsoes.items():
            linhas.append({"grupo": grupo, "modelo": nome, **metricas(y[mascara], np.asarray(p)[mascara])})
    return pd.DataFrame(linhas)
