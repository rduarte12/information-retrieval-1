import nltk
# sempre normalizar para minusculas
# funcao de tokenizacao d etexto 

def _tokenize(text: str) -> list: 
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return tokens

# funcao de remocao de stop words
def _remove_stopwords(tokens: list) -> list:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

# funcao de stemming
def _stem(tokens: list) -> list:
    stemmer = nltk.stem.PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# funcao de preprocessamento
def preprocess(text: str, remove_stopwords: bool = True, stem: bool = True) -> list:
    tokens = _tokenize(text)
    if remove_stopwords:
        tokens = _remove_stopwords(tokens)
    if stem:
        tokens = _stem(tokens)
    return tokens
