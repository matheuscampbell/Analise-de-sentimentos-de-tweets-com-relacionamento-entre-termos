import time

import tweetcon as tc


def get_tweet_rts(tweet, max=10):
    lista_tweets = []
    for t in tweet:
        rtwets = tc.get_retweets_by_tweet(t.id, max)
        time.sleep(1)
        if len(rtwets) > 0:
            for rtw in rtwets:
                rtw.tipo = 'retweet'
                rtw.retweet_from = t.user.screen_name
                rtw.retweet_from_id = t.user.id
                lista_tweets.append(rtw)
    return lista_tweets
