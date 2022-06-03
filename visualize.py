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
