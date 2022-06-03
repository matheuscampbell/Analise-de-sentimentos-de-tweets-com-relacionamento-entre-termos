import nltk
import string
import heapq
nltk.download('punkt')


def resumodetextoemNpalavras(texto, n):
    # Remover pontuação
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Tokenizar texto
    tokens = nltk.word_tokenize(texto)
    # Tabela de frequência
    freq = nltk.FreqDist(tokens)
    # Lista de tuplas (palavra, frequência)
    tuplas = freq.most_common(n)
    # Lista de palavras
    palavras = [p[0] for p in tuplas]
    # Texto resumido
    textoresumido = ' '.join(palavras)
    return textoresumido

def get_resum(texto):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    # Remover pontuação
    pontuation = string.punctuation


def processamento(texto):
    texto_formatado = texto.lower()
    tokens = []
