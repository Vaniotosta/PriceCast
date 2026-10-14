from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DADOS_BRUTOS = RAIZ / "data" / "raw" / "online_retail_II.xlsx"
DADOS_LIMPOS = RAIZ / "data" / "processed" / "vendas_limpas.csv.gz"
MODELO = RAIZ / "models" / "pricecast.joblib"
CATALOGO = RAIZ / "models" / "catalogo.csv"
METRICAS = RAIZ / "reports" / "metricas.json"

# Teste = últimos meses da base (outubro a dezembro de 2011)
DATA_CORTE = "2011-10-01"
DIAS_CALIBRACAO = 60          # janela para calcular as faixas de incerteza
SEMENTE = 42
