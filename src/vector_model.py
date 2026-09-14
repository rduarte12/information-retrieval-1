# src/vector_model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess

class VectorModel:
    def __init__(self, processed_docs: dict, processed_queries: dict):
        """
        processed_docs:    {doc_id -> list[str]} tokens ja  processados
        processed_queries: {query_id -> list[str]} — tokens processados
        """
        self.doc_ids   = list(processed_docs.keys())
        self.query_ids = list(processed_queries.keys())

        # converte list[str] => str para o vectorizer (que vai apenas splittar)
        corpus  = [" ".join(tokens) for tokens in processed_docs.values()]
        queries = [" ".join(tokens) for tokens in processed_queries.values()]

        # como o corpus ja esta processado apenas split no espaco
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,  # split no espaco tokens ja estao prontos
            token_pattern=None,  
            lowercase=False       # ja esta em minusculas
        )

        self.doc_matrix   = self.vectorizer.fit_transform(corpus)   # (n_docs, vocab)
        self.query_matrix = self.vectorizer.transform(queries)      # (n_queries, vocab)

    def rank(self, query_id: str) -> list[tuple]:
        q_idx = self.query_ids.index(query_id)
        q_vec = self.query_matrix.getrow(q_idx)
        scores = cosine_similarity(q_vec, self.doc_matrix).flatten()
        return sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
    
    def search(self, raw_text: str, remove_stopwords: bool, stem: bool) -> list[tuple]:
        tokens = preprocess(raw_text, remove_stopwords=remove_stopwords, stem=stem)
        q_vec  = self.vectorizer.transform([" ".join(tokens)])
        scores = cosine_similarity(q_vec, self.doc_matrix).flatten()
        return sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)