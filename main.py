import tweetcon as tc
import analysis as an
import cloudWords as cw
import visualize as vis
import auxiliar as aux

# myKeys = open('twitterKeys.txt', 'r').read().splitlines();


def tweet_analysis(numberOfTweets, SearchTerm, search_relation_with_this_terms):

    public_tweets = tc.get_tweets(SearchTerm, numberOfTweets)
    lista_tweets = aux.get_tweet_rts(public_tweets)
    # merge public_tweets and lista_tweets
    tweets = public_tweets + lista_tweets

    # analise = an.openai_sentiment_analysis(public_tweets)
    # analiza os tweets que falam sobre o SearchTerm
    analise = an.sentimentalAnalizerTwitters(tweets)
    # monta um grafico com as emoções
    vis.analizar_resultado(analise)
    # analiza o que cada usoario fala sobre os termos da search_relation_with_this_terms
    result = an.search_relation_and_run_sentiment_analysis(tweets, search_relation_with_this_terms, 2)
    # monta grafo
    vis.monta_grafo(result, SearchTerm, search_relation_with_this_terms)

    # get text from tweets
    text = ""
    for tweet in public_tweets:
        text += " " + an.limpezaDeDados(an.removeStopWords(tweet.text))

    # get cloud words
    cloudWords = cw.create_cloudwords(text)

# main
if __name__ == '__main__':
            # do while loop
    while True:
        termo = input("Digite o nome do termo a ser analisado: ")
        qtd_tweets = int(input("Digite a quantidade de tweets a ser analisado: "))
        search_relation_with_this_terms = input("Digite os termos a serem analisados: ").split(",")
        print("\nvocê vai pesquisar sobre o termo: " + termo + " e analizar " + str(qtd_tweets) + " tweets e relacionar com os seguintes termos: " + str(search_relation_with_this_terms))
        ok = input("\nDeseja continuar? (s/n) ")
        if ok == 's' or ok == 'S':
            print("\nAnalisando...")
            tweet_analysis(qtd_tweets, termo, search_relation_with_this_terms)



