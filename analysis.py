import re

from matplotlib import pyplot as plt
from textblob import TextBlob
from googletrans import Translator
import pandas as pd

import nltk
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('wordnet')

translator = Translator()


def sentimentalAnalizerTwitters(tweets):
    sum_sentiment = 0

    positive_tweets = 0
    negative_tweets = 0
    neutral_tweets = 0

    users_positives = []
    users_negatives = []
    users_neutrals = []

    if len(tweets) > 0:
        for tweet in tweets:
            if tweet.full_text != None:
                tweet.text = tweet.full_text
            # text = tratar(tweet.text)
            text = tweet.text
            translated = translator.translate(text, src='pt', dest='en', group='tts')
            analysis = TextBlob(translated.text)
            sum_sentiment += analysis.sentiment.polarity

            if analysis.sentiment.polarity > 0:
                positive_tweets += 1
                users_positives.append(tweet.user.screen_name)
            elif analysis.sentiment.polarity < 0:
                negative_tweets += 1
                users_negatives.append(tweet.user.screen_name)
            else:
                neutral_tweets += 1
                users_neutrals.append(tweet.user.screen_name)

    if sum_sentiment != 0 and len(tweets) > 0:
        average_sentiment = sum_sentiment / len(tweets)
    else:
        average_sentiment = 0
        info = "Nenhum tweet foi analisado"

    info = "A média de sentimento dos tweets é: " + str(average_sentiment) + "\n"

    return info, average_sentiment, positive_tweets, negative_tweets, neutral_tweets, users_positives, users_negatives, users_neutrals


def removeStopWords(tweetText):
    # remove stop words
    stop_words = nltk.corpus.stopwords.words('portuguese')
    tweetText = ' '.join([word for word in tweetText.split() if word not in stop_words])
    return tweetText


def Stemming(tweetText):
    # stemming
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for word in tweetText.split():
        palavras.append(stemmer.stem(word))
    tweetText = ' '.join(palavras)
    return tweetText


def limpezaDeDados(tweetText):
    # remove links, pontos, virgulas e acentos dos tweets
    tweetText = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweetText, flags=re.MULTILINE)
    return tweetText


def lemmatization(tweetText):
    # lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    palavras = []
    for word in tweetText.split():
        palavras.append(lemmatizer.lemmatize(word))
    tweetText = ' '.join(palavras)
    return tweetText


def tratar(tweet_text):
    tweet_text = limpezaDeDados(tweet_text)
    tweet_text = removeStopWords(tweet_text)
    # tweet_text = Stemming(tweet_text)
    tweet_text = lemmatization(tweet_text)
    return tweet_text


# ----------------------------------------------------------------------------------------------------------------------


def loadDataTainner():
    # load data
    df = pd.read_csv('./files/Tweets_Mg.csv', encoding='utf-8')
    df.drop_duplicates(['Text'], inplace=True)
    tweets_data = df['Text']
    tweets = []
    classes = df['Classificacao']

    for t in tweets_data:
        tweets.append(Preprocessing(t))

    return tweets, classes


def Preprocessing(instancia):
    stemmer = nltk.stem.RSLPStemmer()
    instancia = instancia.lower()
    instancia = marque_negacao(instancia)
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.', '').replace(';', '').replace('-', '').replace(
        ':', '').replace(')', '')
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [stemmer.stem(i) for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))


def tokenize(text):
    tweet_tokenizer = TweetTokenizer()
    return tweet_tokenizer.tokenize(text)


def vectorize_tweets(tweets):
    tweet_tokenizer = TweetTokenizer()
    vectorizer = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize, max_features=98)
    freq_tweets = vectorizer.fit_transform(tweets)
    return freq_tweets


def train_model(freq_tweets, classes):
    model = MultinomialNB()
    model.fit(freq_tweets, classes)
    return model


def execute_model(model, texts):
    freq_testes = vectorize_tweets(texts)
    result = []
    for t, c in zip(texts, model.predict(freq_testes)):
        result.append((t, c))
    return result


def marque_negacao(texto):
    negacoes = ['não', 'not']
    negacao_detectada = False
    resultado = []
    palavras = texto.split()
    for p in palavras:
        p = p.lower()
        if negacao_detectada == True:
            p = p + '_NEG'
        if p in negacoes:
            negacao_detectada = True
        resultado.append(p)
    return (" ".join(resultado))


# ----------------------------------------------------------------------------------------------------------------------

def analizar_tweets(tweets_data):
    tweets_processed = []
    for t in tweets_data:
        if t.full_text is not None:
            t.text = t.full_text
        tweets_processed.append(Preprocessing(t.text))

    [tweets, classes] = loadDataTainner()
    tweets = vectorize_tweets(tweets)
    model = train_model(tweets, classes)
    # executa o modelo
    result = execute_model(model, tweets_processed)
    for i in range(len(result)):
        tweets_data[i].sentiment = result[i][1]

    return tweets_data


def analizar_resultado(tweets_processed):
    positivos = 0
    negativos = 0
    neutros = 0
    for t in tweets_processed:
        if t.sentiment == 'Positivo':
            positivos += 1
        elif t.sentiment == 'Negativo':
            negativos += 1
        else:
            neutros += 1

    labels = 'Positivos', 'Negativos', 'Neutros'
    sizes = [positivos, negativos, neutros]
    colors = ['gold', 'yellowgreen', 'lightcoral']
    explode = (0.1, 0.1, 0.1)
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()
