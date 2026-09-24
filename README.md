# information-retrieval-1

Trabalho Prático 1 — SCC0282 Recuperação de Informação (USP/ICMC, 2º sem/2026). Sistema de
recuperação textual clássico sobre a coleção Cranfield, comparando o Modelo Vetorial (TF-IDF +
similaridade do cosseno) e o Modelo Probabilístico BM25, com
avaliação quantitativa (Precision@10, Recall@10, MAP, F1@10, MRR, NDCG@10).

## Integrantes

- Rafael Mendonça Duarte - rmduarte@usp.br
- Joao pedro Correia Caetano - joaopcaetanoc@usp.br

## Base de dados

Coleção **Cranfield** (1400 documentos, 225 consultas, 1837 julgamentos de relevância), obtida
via a biblioteca [`ir_datasets`][ir-datasets-cranfield]
(`ir_datasets.load("cranfield")`). Os dados são
obtidos automaticamente na primeira execução e ficam em cache local
(`~/.ir_datasets/cranfield/`).

## Estrutura do projeto

```text
src/
  data_loader.py    - carregamento da coleção Cranfield via ir_datasets
  preprocessing.py  - tokenização, lowercase, stopwords, stemming (Porter)
  vector_model.py   - Modelo Vetorial (TF-IDF + similaridade do cosseno)
  bm25.py           - Modelo Probabilístico BM25 (fórmula implementada manualmente)
  evaluation.py     - métricas de avaliação (binarização de qrels, P@10, R@10, AP/MAP,
                      F1@10, RR/MRR, NDCG@10)
  main.ipynb        - notebook principal: executa e documenta todos os experimentos
  results/
    corpora/        - corpora pré-processados (4 configurações x documentos/consultas)
    rankings/       - rankings completos por modelo e configuração
    per_query/      - métricas por consulta (uma linha por query, todas as métricas)
    aggregated/     - métricas agregadas, comparação entre modelos, grid de parâmetros do
                      BM25, estudos de caso por consulta, reformulação de consultas e
                      análise de erros
requirements.txt
```

## Instalação e execução

### Local

Requer **Python 3.14** (desenvolvido e testado com 3.14.6).

```bash
git clone https://github.com/rduarte12/information-retrieval-1.git
cd information-retrieval-1
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

Abra `src/main.ipynb` (Jupyter, JupyterLab ou VS Code) e execute todas as células em ordem
("Run All"). A primeira célula detecta automaticamente que não está no Colab e apenas confirma
o diretório de trabalho atual; a célula seguinte baixa os recursos do NLTK necessários (`punkt`,
`stopwords`, `punkt_tab`). Os resultados são gravados em `src/results/` (caminho relativo ao
próprio notebook).

### Google Colab

Abra o notebook diretamente a partir do GitHub:

```text
https://colab.research.google.com/github/rduarte12/information-retrieval-1/blob/main/src/main.ipynb
```

A primeira célula detecta o ambiente Colab automaticamente, clona o repositório, entra em
`information-retrieval-1/src` e instala as dependências de `requirements.txt` e rodar.

## Versão da linguagem e principais bibliotecas

- **Python 3.14**
- `ir_datasets` — carregamento da coleção Cranfield (documentos, consultas, qrels)
- `nltk` — tokenização, stopwords (inglês) e stemming (Porter)
- `scikit-learn` — `TfidfVectorizer` e `cosine_similarity` (Modelo Vetorial)
- `pandas` / `numpy` — manipulação de dados e métricas
- `matplotlib` — gráficos e heatmaps

Versões exatas em [`requirements.txt`](requirements.txt).

## Resultados

Todos os resultados dos experimentos estão versionados em `src/results/`:

- `results/corpora/` — corpora pré-processados nas 4 configurações (sem stopwords/sem stemming,
  com stopwords, com stemming, com stopwords+stemming)
- `results/rankings/` — rankings completos (todas as consultas x todos os documentos) por
  modelo e configuração
- `results/per_query/` — métricas por consulta (P@10, R@10, AP, F1@10, RR, NDCG@10)
- `results/aggregated/` — métricas agregadas (`metrics_summary.csv`), comparação entre modelos
  (`model_comparison.csv`, `per_query_diff.csv`, `metrics_by_config.png`), estudos de caso por
  consulta (`query_case_studies.csv`), grid de parâmetros do BM25
  (`bm25_param_grid.csv`, `bm25_param_heatmap.png`, `bm25_b_param_query_comparison.csv`),
  reformulação de consultas (`query_reformulation.csv`, `query_reformulation_summary.csv`) e
  análise de erros (`error_analysis.csv`)

## Uso de ferramentas de IA

IA foi utilizado como apoio ao desenvolvimento, principamente na depuração. Detalhes na seção "Uso de ferramentas de IA" do
relatório final.

[ir-datasets-cranfield]: https://ir-datasets.com/cranfield.html
