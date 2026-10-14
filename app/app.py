"""Painel PriceCast (Streamlit). Rodar com:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from pricecast.config import CATALOGO, DADOS_LIMPOS, METRICAS, MODELO
from pricecast.modelos import ModeloPriceCast

st.set_page_config(page_title="PriceCast", page_icon="💷", layout="wide")


@st.cache_resource
def carregar_modelo():
    return ModeloPriceCast.carregar(MODELO)


@st.cache_data
def carregar_catalogo():
    return pd.read_csv(CATALOGO, dtype={"StockCode": str})


@st.cache_data
def carregar_historico(codigo):
    if not DADOS_LIMPOS.exists():
        return None
    df = pd.read_csv(DADOS_LIMPOS, usecols=["StockCode", "InvoiceDate", "Price"],
                     dtype={"StockCode": str}, parse_dates=["InvoiceDate"])
    return df[df.StockCode == codigo]


if not MODELO.exists():
    st.error("Modelo não encontrado. Rode primeiro: `python scripts/treinar.py`")
    st.stop()

modelo, catalogo = carregar_modelo(), carregar_catalogo()
st.title("💷 PriceCast — Previsão de Preços")
st.caption("Projeto educacional e de portfólio. Não é recomendação comercial ou financeira.")

st.markdown("""
### 🎯 O problema de negócio
Uma loja online de presentes e decoração vende milhares de produtos para clientes do varejo e do atacado.
Na hora de montar um orçamento, cadastrar um produto novo ou revisar a tabela de preços, a equipe comercial
precisa responder rápido: **"qual preço é razoável para este item, nesta quantidade, nesta época?"**

Sem apoio, essa resposta vem de planilhas, memória ou tentativa e erro, e um preço mal estimado
significa **vender barato demais** (perda de margem) ou **caro demais** (perda de venda).

### 💡 O que o PriceCast entrega
- **Um preço estimado** a partir do histórico de vendas de dois anos (cerca de 1 milhão de linhas).
- **Uma faixa provável (80%)**, para mostrar o quanto a estimativa é confiável.
- **Um aviso claro para produtos novos**, que têm menos informação e portanto mais incerteza.
""")

with st.expander("📖 Como ler os resultados e quais são os limites"):
    st.markdown("""
- **Produto conhecido** (já foi vendido antes): o modelo usa o histórico de preços dele. É o caso mais confiável.
- **Produto novo** (sem vendas anteriores): o modelo só tem a descrição e a quantidade. A previsão serve como
  ponto de partida, **não como decisão final**. Veja sempre a faixa provável.
- Os preços da base são **preços praticados em vendas passadas**, não o preço ideal. O modelo não considera
  custo, margem, concorrência ou promoções.
- A faixa de 80% acerta em cerca de **3 de cada 4** vendas no teste, então ela ajuda, mas não é garantia.
""")

aba1, aba2, aba3 = st.tabs(["🔮 Prever preço", "📈 Histórico do produto", "📊 Desempenho do modelo"])

with aba1:
    modo = st.radio("Tipo de produto", ["Produto do catálogo", "Produto novo (só descrição)"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    quantidade = c1.number_input("Quantidade", min_value=1, value=12, step=1)
    pais = c2.selectbox("País", ["United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Other"])
    data = c3.date_input("Data da venda", value=pd.Timestamp("2011-12-10"))

    if modo == "Produto do catálogo":
        catalogo["rotulo"] = catalogo["StockCode"] + " — " + catalogo["Description"]
        escolha = st.selectbox("Produto", catalogo["rotulo"])
        codigo = escolha.split(" — ")[0]
        descricao = catalogo.loc[catalogo.StockCode == codigo, "Description"].iloc[0]
    else:
        codigo = "00000"
        descricao = st.text_input("Descrição do produto (em inglês, como na base)", "WOODEN CHEST OF 6 DRAWERS")

    if st.button("Prever", type="primary"):
        entrada = pd.DataFrame([{"StockCode": codigo, "Description": descricao.upper(),
                                 "Quantity": quantidade, "Country": pais,
                                 "InvoiceDate": pd.Timestamp(data) + pd.Timedelta(hours=12)}])
        r = modelo.prever(entrada).iloc[0]
        a, b = st.columns(2)
        a.metric("Preço previsto", f"£ {r.preco_previsto:,.2f}")
        b.metric("Faixa provável (80%)", f"£ {r.faixa_min:,.2f} – £ {r.faixa_max:,.2f}")
        if r.tipo == "inedito":
            st.warning("Produto inédito: a previsão usa só texto e quantidade, então a incerteza é bem maior.")
        else:
            st.success("Produto conhecido: previsão baseada no histórico de preços dele.")

with aba2:
    escolha2 = st.selectbox("Produto", catalogo["StockCode"] + " — " + catalogo["Description"], key="hist")
    hist = carregar_historico(escolha2.split(" — ")[0])
    if hist is None or hist.empty:
        st.info("Histórico indisponível. Rode `python scripts/preparar_dados.py`.")
    else:
        st.plotly_chart(px.scatter(hist, x="InvoiceDate", y="Price", title="Preços observados nas vendas"),
                        width="stretch")
with aba3:
    if METRICAS.exists():
        try:
            conteudo_metricas = METRICAS.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            conteudo_metricas = METRICAS.read_text(encoding="cp1252")

        try:
            m = json.loads(conteudo_metricas)

            st.write(
                f"Teste a partir de **{m['data_corte']}** — "
                f"{m['registros_teste']:,} vendas "
                f"({m['produtos_ineditos_no_teste']} produtos inéditos)."
            )

            tab = pd.DataFrame(m["tabela"])
            st.dataframe(tab.round(3), width="stretch")

            graf = px.bar(
                tab,
                x="grupo",
                y="MAE",
                color="modelo",
                barmode="group",
                title="Erro médio (MAE, em £) — menor é melhor",
            )
            st.plotly_chart(graf, width="stretch")

            st.caption(f"Cobertura da faixa 80%: {m['cobertura_faixa']}")

        except (json.JSONDecodeError, KeyError, TypeError) as erro:
            st.error(f"Não foi possível interpretar as métricas: {erro}")
    else:
        st.info("Rode `python scripts/treinar.py` para gerar as métricas.")