import tweepy


def auth():
    api_key = '4jJsXVL2G1b83MMZsa6pcEoqj'
    api_secret = 'cKHIjOBl4nJYJ4W8xWIuvNUDdE9J95zwdUFmuMO5Sn2Em8B8af'

    consumer_key = 'Q2FPZTM0cGt1Q0pKVVIwbjdUR2c6MTpjaQ'
    consumer_secret = 'wxHTIlOvAjltTzGD4U_pkuWzECQh5KnHW0KFW-qVZW876Ve_tu'
    access_token = '1529939844847353873-vPj5mevOElczdPYvru3tbcezS5ZigP'
    access_token_secret = '6XKbYqbXhQgva6TbZFLjDaaCF4tFvdfwfXQ5wNgmGgPTm'
    barbertoken = ''
    # read from file
    barbertoken = open('barber.txt', 'r').read()
    auth = tweepy.OAuth2BearerHandler(barbertoken)

    return tweepy.API(auth)


def get_tweets(query, qtd_tweets=100):
    api = auth()
    r = api.search_tweets(q=query, count=qtd_tweets, lang='pt', result_type='popular', tweet_mode='extended',
                             include_rts=False)
    for t in r:
        t.tipo = 'tweet'

    return r


def get_tweets_by_user(user, qtd_tweets=100):
    api = auth()
    return api.user_timeline(screen_name=user, count=qtd_tweets, lang='pt', result_type='recent', include_rts=True,
                             tweet_mode='extended')


def get_user(user):
    api = auth()
    return api.get_user(screen_name=user)


def search_Term(term, tweets):
    founds = []
    term = term.lower()
    for tweet in tweets:
        if term in tweet.full_text.lower():
            founds.append(tweet)

    return founds


def search_tweets_in_user_timeline(user, term, qtd_tweets=100):
    tweets = get_tweets_by_user(user, qtd_tweets)
    founds = search_Term(term, tweets)
    return founds


def get_retweets_by_tweet(id, max=10):
    api = auth()
    return api.get_retweets(id, count=max)


def get_trends():
    api = auth()
    return api.get_place_trends(id=23424768)
