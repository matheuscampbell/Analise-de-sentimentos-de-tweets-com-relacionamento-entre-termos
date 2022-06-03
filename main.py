import pandas as pd
import numpy as np
import tweetcon as tc
import analysis as an
import cloudWords as cw
import resumodetexto as rt
import re

# myKeys = open('twitterKeys.txt', 'r').read().splitlines();


# main
if __name__ == '__main__':
    # get tweets
    numberOfTweets = 1000


    public_tweets = tc.search_tweets_in_user_timeline('LacerdaGeovane', 'bolsonaro', numberOfTweets)
    analise = an.analizar_tweets(public_tweets)
    # print(analise)
    an.analizar_resultado(analise)

    exit(0)

    # get text from tweets
    text = ""
    for tweet in public_tweets:
        text += " " + an.limpezaDeDados(an.removeStopWords(tweet.text))

    # get cloud words
    cloudWords = cw.create_cloudwords(text)

    # get resume
    resume = rt.resumodetextoemNpalavras(text, 50)
    print(resume)
