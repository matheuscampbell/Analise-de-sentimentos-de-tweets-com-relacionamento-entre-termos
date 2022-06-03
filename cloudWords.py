from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

STOPWORDS = ['ver', 'principal', 'essa', 'vez', 'nas', 'mas', 'qual', 'principal', 'ele', 'ter', 'doença', 'pois',
             'este', 'vez', 'ver principal', 'artigo principal', 'já', 'aos', 'pode', 'outro', 'artigo', 'desse',
             'alguns', 'meio', 'entre', 'das', 'podem', 'esse', 'seu', 'também', 'são', 'quando', 'de', 'que', 'em',
             'os', 'as', 'da', 'como', 'dos', 'ou', 'se', 'um', 'uma', 'para', 'na', 'ao', 'mais', 'por', 'não',
             'ainda', 'muito', 'sua', 'https', 'foi', 'nem', 'você', 'vai', 'nosso', 'RT', 't', 'co', 'é', 'e'] + list(STOPWORDS)


def create_cloudwords(text):
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=1920, height=1080).generate(text)
    plt.figure(figsize=(10, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
