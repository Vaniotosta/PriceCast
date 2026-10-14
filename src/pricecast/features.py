"""Variáveis do modelo. Regra de ouro: cada linha só enxerga o PASSADO do produto."""
import numpy as np
import pandas as pd

ESTAT = ["n_prev", "lp_mean", "lp_std", "lp_last", "lp_min", "lp_max", "lp_med15", "lq_mean"]
VARS_QUENTE = ESTAT + ["dias_ult", "idade", "dq", "lq", "mes", "dsem", "dmes", "hora", "uk"]


def calendario(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["mes"] = d["InvoiceDate"].dt.month
    d["dsem"] = d["InvoiceDate"].dt.dayofweek
    d["dmes"] = d["InvoiceDate"].dt.day
    d["hora"] = d["InvoiceDate"].dt.hour
    d["uk"] = (d["Country"] == "United Kingdom").astype(int)
    d["lq"] = np.log(d["Quantity"])
    return d


def historico_por_linha(d: pd.DataFrame) -> pd.DataFrame:
    """Para o TREINO: estatísticas do produto usando só vendas anteriores a cada linha."""
    d = d.sort_values("InvoiceDate").copy()
    d["lp"] = np.log(d["Price"])
    d["lq"] = np.log(d["Quantity"])
    d["lp2"] = d["lp"] ** 2
    g = d.groupby("StockCode", sort=False)
    d["n_prev"] = g.cumcount()
    n = d["n_prev"].replace(0, np.nan)
    soma = g["lp"].cumsum() - d["lp"]
    soma2 = g["lp2"].cumsum() - d["lp2"]
    d["lp_mean"] = soma / n
    d["lp_std"] = np.sqrt(np.maximum(soma2 / n - d["lp_mean"] ** 2, 0))
    d["lp_last"] = g["lp"].shift(1)
    d["lp_min"] = g["lp"].cummin().groupby(d["StockCode"]).shift(1)
    d["lp_max"] = g["lp"].cummax().groupby(d["StockCode"]).shift(1)
    d["_ant"] = d["lp_last"]
    d["lp_med15"] = (d.groupby("StockCode", sort=False)["_ant"]
                      .transform(lambda s: s.rolling(15, min_periods=1).median()))
    d["lq_mean"] = (g["lq"].cumsum() - d["lq"]) / n
    primeira = g["InvoiceDate"].transform("min")
    anterior = g["InvoiceDate"].shift(1)
    d["dias_ult"] = (d["InvoiceDate"] - anterior).dt.total_seconds() / 86400
    d["idade"] = (d["InvoiceDate"] - primeira).dt.total_seconds() / 86400
    d["dq"] = d["lq"] - d["lq_mean"]
    return calendario(d.drop(columns=["lp2", "_ant"]))


def tabela_produtos(treino: pd.DataFrame) -> pd.DataFrame:
    """Resumo final de cada produto no treino (usado para prever o futuro)."""
    d = treino.sort_values("InvoiceDate").copy()
    d["lp"] = np.log(d["Price"])
    d["lq"] = np.log(d["Quantity"])
    g = d.groupby("StockCode")
    return pd.DataFrame({
        "n_prev": g.size(), "lp_mean": g["lp"].mean(), "lp_std": g["lp"].std(ddof=0),
        "lp_last": g["lp"].last(), "lp_min": g["lp"].min(), "lp_max": g["lp"].max(),
        "lp_med15": g["lp"].apply(lambda s: s.iloc[-15:].median()),
        "lq_mean": g["lq"].mean(),
        "t_first": g["InvoiceDate"].min(), "t_prev": g["InvoiceDate"].max(),
        "Description": g["Description"].last(),
        "preco_mediano": g["Price"].median(),
    })


def aplicar_tabela(novos: pd.DataFrame, tabela: pd.DataFrame) -> pd.DataFrame:
    """Para o TESTE / uso real: junta o resumo do produto (congelado no treino)."""
    t = tabela.reindex(novos["StockCode"].values)
    t.index = novos.index
    d = calendario(novos)
    for c in ESTAT:
        d[c] = t[c]
    d["n_prev"] = d["n_prev"].fillna(0)
    d["dias_ult"] = (d["InvoiceDate"] - t["t_prev"]).dt.total_seconds() / 86400
    d["idade"] = (d["InvoiceDate"] - t["t_first"]).dt.total_seconds() / 86400
    d["dq"] = d["lq"] - d["lq_mean"]
    return d
