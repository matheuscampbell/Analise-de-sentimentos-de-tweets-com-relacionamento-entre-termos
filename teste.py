import grafo


if __name__ == '__main__':
    g = grafo.Grafo()
    g.newNode(1, 'Bruno Araújo', 'Bruno Araújo', 'square', '#FF0000', 10)
    g.newNode(2, 'Dom Phillips', 'Dom Phillips', 'square', '#FF0000', 10)
    g.newNode(3, 'Bolsonaro', 'Bolsonaro', 'square', '#FF0000', 10)
    g.newNode(4, 'Lula', 'Lula', 'square', '#FF0000', 10)
    g.newEdge(1, 2, '', 1)
    g.newEdge(1, 3, '', 1)
    g.newEdge(1, 4, '', 1)
    g.newEdge(2, 3, '', 1)
    g.newEdge(2, 4, '', 1)
    g.createNetworkGraph()
