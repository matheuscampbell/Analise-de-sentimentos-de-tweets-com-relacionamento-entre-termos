import json

from matplotlib import pyplot as plt
from pyvis.network import Network
import networkx as nx


def createGraph(nodes, edges):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    return G


def createNetwork(Graph):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    for node in Graph.nodes():
        net.add_node(node, label=node, title=node, shape="circle", color="#00ff00", size=10)
    for edge in Graph.edges():
        net.add_edge(edge[0], edge[1])
    return net


def drawGraph(Graph):
    net = createNetwork(Graph)
    net.show("graph.html")


def createNetworkGraph(nodes, edges):
    graph = createGraph(nodes, edges)
    net = createNetwork(graph)
    drawGraph(net)


def test():
    nx_graph = nx.Graph()
    nx_graph.add_node(20, size=20, title='couple', group=2)
    nx_graph.add_node(21, size=15, title='couple', group=2)
    nx_graph.add_edge(20, 21, weight=5)
    nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
    nx_graph.add_edge(20, 25, weight=5)
    nt = Network('500px', '500px')
    nt.from_nx(nx_graph)
    nt.show('nx.html')


def monta_grafo(analysis_result, serached_term, terms_to_relationize):
    color = ''
    nx_graph = nx.Graph()
    nx_graph.add_node(0, size=40, title=serached_term, label=serached_term, group=0, x=1500, y=1850)
    trmcount = 1
    users_processed = []
    for term in terms_to_relationize:
        nx_graph.add_node(trmcount, size=40, title=term, label=term, group=trmcount,
                          x=(800 / len(terms_to_relationize)) + trmcount * 100, y=150)
        trmcount += 1
    for t in analysis_result:
        print(t['sentiment'], t['term'])
        i = terms_to_relationize.index(t['term']) + 1
        id = t['userid']

        if t['frist_sentiment'] == 'Positivo':
            color = '#00ff00'
        elif t['frist_sentiment'] == 'Negativo':
            color = '#ff0000'
        elif t['frist_sentiment'] == 'Neutro':
            color = '#ffff00'

        if t['tipo'] == 'retweet':
            if id not in users_processed:
                nx_graph.add_node(id, size=27, title=t['user'] + ' - ' + t['tweet'], image=t['img'], group=i,
                                  color=color)
            nx_graph.add_edge(t['retweet_from_id'], id, weight=t['retweets'], color=color)
        else:
            if id not in users_processed:
                nx_graph.add_node(id, size=35, title=t['user'] + ' - ' + t['tweet'], image=t['img'], group=i,
                                  color=color)
            nx_graph.add_edge(0, id, weight=t['retweets'], color=color)

        users_processed.append(t['userid'])

        if t['sentiment'] == 'Positivo':
            color = '#00ff00'
        elif t['sentiment'] == 'Negativo':
            color = '#ff0000'
        elif t['sentiment'] == 'Neutro':
            color = '#ffff00'

        nx_graph.add_edge(i, id, weight=t['retweets'], color=color)




    nt = Network('1000px', '1000px')
    nt.from_nx(nx_graph)
    nt.show_buttons(filter_=['physics'])
    # nt.set_options('var options = {"physics": {"barnesHut": {"theta": 0.65,"gravitationalConstant": -73902,"centralGravity": 0,"springLength": 500,"springConstant": 0.005,"damping": 0.14,"avoidOverlap": 0.34 },"minVelocity": 0.75}}')
    nt.show('nx.html')


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
