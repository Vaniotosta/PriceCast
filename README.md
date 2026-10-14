# PriceCast — Previsão de Preços e Inteligência Comercial
lINK DO APP: https://pricecast-jq6cw55ww3jsf2nc4btquo.streamlit.app

Projeto de Ciência de Dados que prevê o preço unitário de produtos a partir de vendas históricas
(*Online Retail II*, UCI) e mostra os resultados em um painel Streamlit.

> Finalidade educacional e de portfólio. As previsões não são garantia de preços futuros nem recomendação comercial ou financeira.

## O que mudou na versão 2

| Problema da v1 | Solução na v2 |
|---|---|
| Usava só a aba 2009-2010 (≈ 500 mil linhas) | Usa **as duas abas** (≈ 1 milhão de linhas, 2 anos) |
| `StockCode` em one-hot: produto novo ficava sem informação | **Dois modelos**: *quente* (produto conhecido, usa o histórico de preços) e *frio* (produto novo, usa texto + quantidade) |
| Previa o preço direto (muito assimétrico) | Prevê o **log do preço** e o desvio em relação ao preço recente do produto |
| Dava um número só | Dá o preço **e uma faixa provável (80%)** |
| Um único R² misturava cenários | Avaliação **separada**: produtos conhecidos × inéditos, sempre contra uma referência simples |

## Resultados (teste: 01/10/2011 a 09/12/2011, treino com tudo antes disso)

| Grupo | Modelo | MAE (£) | RMSE (£) | R² |
|---|---|---|---|---|
| Todos | Mediana por produto (referência) | 0,785 | 2,091 | 0,633 |
| Todos | **PriceCast v2** | **0,694** | **1,630** | **0,777** |
| Conhecidos | Mediana por produto | 0,663 | 1,767 | 0,723 |
| Conhecidos | **PriceCast v2** | **0,598** | **1,343** | **0,840** |
| Inéditos | Mediana por produto* | 3,215 | 5,405 | −0,346 |
| Inéditos | **PriceCast v2** | **2,614** | **4,426** | **0,097** |

\* Para produto inédito não existe mediana; a referência usa a mediana global.

**Leitura honesta dos números**
- A maioria dos produtos mantém o mesmo preço por meses. Por isso a mediana por produto acerta *exatamente* na maior parte das vendas (erro percentual mediano 0%), enquanto o modelo erra ~10% no caso típico. O modelo ganha porque erra **muito menos nos casos de mudança de preço**.
- Produtos inéditos continuam difíceis (R² ≈ 0,10). A faixa de incerteza existe justamente para deixar isso claro no painel.
- A faixa 10%–90% cobre ≈ 74% (conhecidos) e ≈ 72% (inéditos) das vendas de teste; o ideal seria ≈ 80%, então a faixa está um pouco estreita.
- Os números da v1 (`notebooks/02_modelagem_preco_v1.ipynb`) usavam outro corte (out/2010) e só um ano de dados; não são diretamente comparáveis com esta tabela.

## Estrutura

```
pricecast/
├── app/app.py               # painel Streamlit
├── src/pricecast/           # código reutilizável (dados, features, modelos, avaliação)
├── scripts/
│   ├── preparar_dados.py    # Excel -> CSV limpo
│   └── treinar.py           # avalia, treina e salva o modelo
├── notebooks/               # exploração (notebook da v1)
├── tests/                   # pytest
├── data/                    # dados (não vão para o GitHub)
├── models/                  # modelo treinado (~3 MB, vai para o GitHub para o app funcionar online)
├── reports/metricas.json    # métricas do último treino
└── .github/workflows/ci.yml # testes automáticos
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# coloque o Excel em data/raw/ (veja data/README.md)
python scripts/preparar_dados.py
python scripts/treinar.py
streamlit run app/app.py
pytest -q
```

## Próximos passos
- Calibrar a faixa de incerteza (hoje cobre ~73% em vez de 80%).
- Modelo híbrido: manter o preço recente quando o produto é estável e usar o modelo só quando houver sinal de mudança.
- Embeddings de texto (em vez de TF-IDF) para produtos inéditos.
- Backtesting em várias janelas de tempo e acompanhamento com MLflow.
- Validação de dados com Pandera.

## Fonte dos dados
Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii
