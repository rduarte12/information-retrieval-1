# src/evaluation.py

import math
import pandas as pd
from collections import defaultdict


def binarize_qrels(qrels: list) -> dict:
    """
    qrels: [(query_id, doc_id, relevancia), ...]
    retorna: {query_id -> set(doc_ids com relevancia >= 1)}
    """
    relevant = defaultdict(set)
    for query_id, doc_id, relevance in qrels:
        if relevance >= 1:
            relevant[query_id].add(doc_id)
    return dict(relevant)


def grade_qrels(qrels: list) -> dict:
    """
    Para NDCG — mantem graus positivos originais.
    retorna: {query_id -> {doc_id -> grau}}
    """
    grades = defaultdict(dict)
    for query_id, doc_id, relevance in qrels:
        if relevance >= 1:
            grades[query_id][doc_id] = relevance
    return dict(grades)


# --- metricas por query ---

def precision_at_k(ranking: list, relevant: set, k=10) -> float:
    top_k = [doc_id for doc_id, _ in ranking[:k]]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(ranking: list, relevant: set, k=10) -> float:
    if not relevant:
        return 0.0
    top_k = [doc_id for doc_id, _ in ranking[:k]]
    hits  = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def average_precision(ranking: list, relevant: set) -> float:
    if not relevant:
        return 0.0
    hits, total, ap = 0, 0, 0.0
    for doc_id, _ in ranking:
        total += 1
        if doc_id in relevant:
            hits += 1
            ap   += hits / total
    return ap / len(relevant)


def f1_at_k(ranking: list, relevant: set, k=10) -> float:
    p = precision_at_k(ranking, relevant, k)
    r = recall_at_k(ranking, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def reciprocal_rank(ranking: list, relevant: set) -> float:
    for i, (doc_id, _) in enumerate(ranking, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list, grades: dict, k=10) -> float:
    def dcg(docs):
        return sum(
            grades.get(doc_id, 0) / math.log2(i + 2)
            for i, doc_id in enumerate(docs)
        )

    top_k  = [doc_id for doc_id, _ in ranking[:k]]
    ideal  = sorted(grades.keys(), key=lambda d: grades[d], reverse=True)[:k]

    dcg_val  = dcg(top_k)
    idcg_val = dcg(ideal)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0


# --- avaliacao completa ---

def evaluate(rankings: dict, qrels: list) -> pd.DataFrame:
    """
    rankings: {query_id -> [(doc_id, score), ...]}
    qrels:    [(query_id, doc_id, relevancia), ...]
    retorna:  DataFrame com uma linha por query e todas as metricas
    """
    relevant_bin = binarize_qrels(qrels)
    relevant_grad = grade_qrels(qrels)

    rows = []
    for query_id, ranking in rankings.items():
        relevant = relevant_bin.get(query_id, set())
        grades   = relevant_grad.get(query_id, {})

        rows.append({
            "query_id":  query_id,
            "P@10":      precision_at_k(ranking, relevant),
            "R@10":      recall_at_k(ranking, relevant),
            "AP":        average_precision(ranking, relevant),
            "F1@10":     f1_at_k(ranking, relevant),
            "RR":        reciprocal_rank(ranking, relevant),
            "NDCG@10":   ndcg_at_k(ranking, grades),
        })

    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> dict:
    """
    recebe o DataFrame de evaluate() e retorna metricas agregadas.
    MAP = media das AP, MRR = media dos RR, resto e media direta.
    """
    return {
        "MAP":       df["AP"].mean(),
        "MRR":       df["RR"].mean(),
        "P@10":      df["P@10"].mean(),
        "R@10":      df["R@10"].mean(),
        "F1@10":     df["F1@10"].mean(),
        "NDCG@10":   df["NDCG@10"].mean(),
    }