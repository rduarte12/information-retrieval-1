import ir_datasets

# funcao para carregar o dataset a ser usando para avaliare metodos de recuperação da informação

def load_cranfield():
    dataset = ir_datasets.load("cranfield")
    
    # 1. documentos: dict {doc_id > texto}
    docs = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
    
    # 2. consultas: dict {query_id > texto}
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    
    # 3. qrels: lista de tuplas (query_id, doc_id, relevancia)
    qrels = [(qrel.query_id, qrel.doc_id, qrel.relevance) 
             for qrel in dataset.qrels_iter()]
    
    return docs, queries, qrels