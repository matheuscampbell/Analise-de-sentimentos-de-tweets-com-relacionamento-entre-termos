import re
import time

from matplotlib import pyplot as plt
from textblob import TextBlob
from googletrans import Translator
import pandas as pd

import openai

import nltk
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

from tweetcon import search_tweets_in_user_timeline

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('rslp')
nltk.download('stopwords')
nltk.download('wordnet')

translator = Translator()


def sentimentalAnalizerTwitters(tweets):
    if len(tweets) > 0:
        i = 0
        for tweet in tweets:
            if hasattr(tweet, 'full_text'):
                tweet.text = tweet.full_text
            text = Preprocessing(tweet.text)
            translated = translator.translate(text, src='pt', dest='en', group='tts')
            analysis = TextBlob(translated.text)
            if analysis.sentiment.polarity > 0:
                tweets[i].sentiment = 'Positivo'
            elif analysis.sentiment.polarity < 0:
                tweets[i].sentiment = 'Negativo'
            else:
                tweets[i].sentiment = 'Neutro'
            i += 1
    return tweets


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
    # remove @user
    tweetText = re.sub(r'@\w+', '', tweetText, flags=re.MULTILINE)
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
    # instancia = ''.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", instancia).split())
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [stemmer.stem(i) for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))


def tokenize(text):
    tweet_tokenizer = TweetTokenizer()
    return tweet_tokenizer.tokenize(text)


def vectorize_tweets(tweets):
    tweet_tokenizer = TweetTokenizer()
    vectorizer = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize, max_features=6)
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


def openai_sentiment_analysis(tweets_data):
    for t in tweets_data:
        if t.full_text is not None:
            t.text = t.full_text
        t.sentiment = get_openai_sentiment(t.text)
        print(t.sentiment + ' - ' + t.text + '\n')

    return tweets_data


def get_openai_sentiment(tweet):
    tweet = Preprocessing(tweet)
    openai.api_key = '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'
    return openai.Completion.create(
        engine="text-davinci-002",
        prompt="Classify the sentiment in these tweets:\n\n1. " + tweet + "\n\nTweet sentiment ratings:",
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    ).choices[0].text.replace('\n\n1. ', '').replace('Negative', 'Negativo').replace('Positive', 'Positivo').replace(
        'Neutral', 'Neutro')


def removeDupicatesUser(tweets_data):
    users = []
    tweets_processed = []
    for t in tweets_data:
        if t.user.screen_name not in users:
            tweets_processed.append(t)
        else:
            users.append(t.user.screen_name)

    return tweets_processed


# procura relação
def search_relation_and_run_sentiment_analysis(tweets_data=[], terms_to_relationize=[],
                                               qtd_tweets_from_user_to_analize=5, user_open_ai=False):
    i = 0
    terms_result = []
    for t in tweets_data:
        tweets_data[i].result_analysis = []
        for term in terms_to_relationize:
            twts = search_tweets_in_user_timeline(t.user.screen_name, term, qtd_tweets_from_user_to_analize)
            time.sleep(1)
            positive = 0
            negative = 0
            neutral = 0
            for twt in twts:
                if twt.full_text is not None:
                    twt.text = twt.full_text

                if user_open_ai:
                    sentiment = get_openai_sentiment(twt.text)
                else:
                    sentiment = sentimentalAnalizerTwitters([twt])[0].sentiment

                if sentiment == 'Positivo':
                    positive += 1
                elif sentiment == 'Negativo':
                    negative += 1
                else:
                    neutral += 1

            if positive > negative and positive > neutral:
                sentiment = 'Positivo'
            elif negative > neutral and negative > positive:
                sentiment = 'Negativo'
            elif (neutral > positive and neutral > negative) or (positive == negative):
                sentiment = 'Neutro'

            results = {
                'term': term,
                'sentiment': sentiment,
                'user': t.user.screen_name,
                'userid': t.user.id,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'total': positive + negative + neutral,
                'tweet': t.text,
                'img': t.user.profile_image_url_https,
                'frist_sentiment': t.sentiment,
                'retweets': t.retweet_count,
                'favorites': t.favorite_count,
                'tipo': t.tipo,
            }
            if t.tipo == 'retweet':
                results['retweet_from'] = t.retweet_from
                results['retweet_from_id'] = t.retweet_from_id

            terms_result.append(results)
        i += 1
    return terms_result
