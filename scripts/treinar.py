"""Passo 2: avalia no período de teste, depois treina o modelo final com tudo."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pricecast.avaliacao import comparar_por_grupo
from pricecast.config import CATALOGO, DADOS_LIMPOS, DATA_CORTE, METRICAS, MODELO
from pricecast.modelos import ModeloPriceCast

if __name__ == "__main__":
    df = pd.read_csv(DADOS_LIMPOS, parse_dates=["InvoiceDate"], dtype={"StockCode": str})
    corte = pd.Timestamp(DATA_CORTE)
    treino, teste = df[df.InvoiceDate < corte], df[df.InvoiceDate >= corte]
    print(f"Treino: {len(treino):,} | Teste: {len(teste):,}")

    # 1) avaliação honesta: treina só com o passado e testa no futuro
    modelo = ModeloPriceCast().fit(treino)
    prev = modelo.prever(teste)
    conhecido = (prev["tipo"] == "conhecido").values
    mediana = teste["StockCode"].map(treino.groupby("StockCode")["Price"].median())
    base = mediana.fillna(treino["Price"].median()).values       # referência simples
    tabela = comparar_por_grupo(teste["Price"].values,
                                {"Mediana por produto (referência)": base,
                                 "PriceCast v2": prev["preco_previsto"].values}, conhecido)
    print(tabela.round(3).to_string(index=False))

    dentro = ((teste["Price"].values >= prev["faixa_min"].values) &
              (teste["Price"].values <= prev["faixa_max"].values))
    cobertura = {"conhecidos": float(dentro[conhecido].mean()), "ineditos": float(dentro[~conhecido].mean())}
    print("Cobertura da faixa 10%-90% (ideal ~80%):", cobertura)

    METRICAS.parent.mkdir(exist_ok=True)
    METRICAS.write_text(json.dumps({
        "data_corte": DATA_CORTE, "registros_treino": len(treino), "registros_teste": len(teste),
        "produtos_ineditos_no_teste": int(teste.loc[~conhecido, "StockCode"].nunique()),
        "tabela": tabela.to_dict(orient="records"), "cobertura_faixa": cobertura,
    }, indent=2, ensure_ascii=False))

    # 2) modelo final com todos os dados
    final = ModeloPriceCast().fit(df)
    final.salvar(MODELO)
    final.tabela[["Description", "preco_mediano", "n_prev"]].rename(
        columns={"n_prev": "vendas"}).reset_index().to_csv(CATALOGO, index=False)
    print("Modelo salvo em", MODELO)
