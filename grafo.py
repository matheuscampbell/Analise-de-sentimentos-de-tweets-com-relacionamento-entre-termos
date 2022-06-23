from pyvis.network import Network
import networkx as nx


class Grafo:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def createGraph(self):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node)
        for edge in self.edges:
            G.add_edge(edge[0], edge[1])
        return G

    def createNetwork(self, Graph):
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
        for node in Graph.nodes():
            net.add_node(node['user'], label=node['label'], title=node['title'], shape=node['shape'],
                         color=node['color'], size=node['size'])
        for edge in Graph.edges():
            net.add_edge(edge[0], edge[1], weight=edge[2])
        return net

    def drawGraph(self, Graph):
        net = self.createNetwork(Graph)
        net.show("graph.html")

    def createNetworkGraph(self):
        graph = self.createGraph()
        net = self.createNetwork(graph)
        self.drawGraph(net)

    def newNode(self, id, label, title, shape, color, size):
        self.nodes.append({'user': id, 'label': label, 'title': title, 'shape': shape, 'color': color, 'size': size})

    def newEdge(self, id1, id2, label, weight):
        self.edges.append({'source': id1, 'target': id2, 'label': label, 'weight': weight})

