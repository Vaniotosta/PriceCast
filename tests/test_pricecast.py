import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pricecast.avaliacao import metricas
from pricecast.dados import limpar
from pricecast.features import historico_por_linha
from pricecast.modelos import ModeloPriceCast


def _vendas(n_dias=120):
    rng = np.random.default_rng(0)
    linhas = []
    produtos = {"10001": ("RED MUG", 2.5), "10002": ("BLUE MUG", 2.6),
                "10003": ("WOODEN CHEST DRAWERS", 40.0), "10004": ("GLASS CANDLE HOLDER", 1.2)}
    for dia in range(n_dias):
        for cod, (desc, p) in produtos.items():
            for _ in range(3):
                linhas.append({"Invoice": f"5{len(linhas)}", "StockCode": cod, "Description": desc,
                               "Quantity": int(rng.integers(1, 24)), "Price": p * rng.choice([1, 1, 1.1]),
                               "InvoiceDate": pd.Timestamp("2011-01-01") + pd.Timedelta(days=dia, hours=9),
                               "Customer ID": 1.0, "Country": "United Kingdom"})
    return pd.DataFrame(linhas)


def test_limpeza_remove_cancelamentos_e_codigos_que_nao_sao_produto():
    df = pd.DataFrame({
        "Invoice": ["1", "C2", "3", "4", "5"], "StockCode": ["85123A", "85123A", "POST", "22423", "22423"],
        "Description": ["A", "A", "B", "C", "C"], "Quantity": [1, -1, 1, 1, 0],
        "Price": [1.0, 1.0, 5.0, 3.0, 3.0], "InvoiceDate": ["2011-01-01"] * 5,
        "Customer ID": [1.0] * 5, "Country": ["United Kingdom"] * 5})
    assert list(limpar(df)["Invoice"]) == ["1", "4"]


def test_historico_nao_vaza_o_futuro():
    f = historico_por_linha(_vendas(10))
    primeira = f[f.StockCode == "10001"].sort_values("InvoiceDate").iloc[0]
    assert primeira["n_prev"] == 0 and np.isnan(primeira["lp_last"])
    # o preço do próprio registro nunca entra na estatística dele
    seg = f[f.StockCode == "10001"].sort_values("InvoiceDate").iloc[1]
    assert seg["n_prev"] == 1


def test_modelo_prevê_conhecido_e_inedito():
    v = _vendas()
    treino, teste = v[v.InvoiceDate < "2011-04-15"], v[v.InvoiceDate >= "2011-04-15"].copy()
    modelo = ModeloPriceCast(componentes_texto=3).fit(treino, calibrar=False)
    novo = teste.iloc[:1].copy()
    novo["StockCode"], novo["Description"] = "99999", "WOODEN CHEST DRAWERS"
    prev = modelo.prever(pd.concat([teste.head(50), novo]))
    assert prev["tipo"].iloc[-1] == "inedito" and (prev["tipo"].iloc[:-1] == "conhecido").all()
    assert metricas(teste.head(50)["Price"], prev["preco_previsto"].iloc[:-1])["MAE"] < 0.5


def test_metricas():
    m = metricas([1, 2, 3], [1, 2, 4])
    assert round(m["MAE"], 3) == 0.333
