import numpy as np 
import matplotlib.pyplot as plt
import itertools

graph_1 = np.array([[-1, 1],
                    [1, -1]])

graph_2 = np.array([[-3, 1, 1, 1],
                    [1, -3, 1, 1],
                    [1, 1, -3, 1],
                    [1, 1, 1, -3]])

graph_2 = np.array([[-3, 1, 1, 1],
                    [1, -3, 1, 1],
                    [1, 1, -3, 1],
                    [1, 1, 1, -3]])

graph_3 = np.array([[-2, 1, 0, 0, 1],
                    [1, -2, 1, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 1, -2, 1],
                    [1, 0, 0, 1, -2]])

graph_4 = np.random.choice([0, 1], size=(10, 10))
graph_4 = np.abs(graph_4 - graph_4.T)
diag = np.sum(graph_4, axis=1)
graph_4 = graph_4 - np.diag(diag)

graph_5 = np.array([[-2, 1, 0, 1, 0, 0, 0, 0],
                    [1, -3, 1, 1, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0, 0, 0, 0],
                    [1, 1, 1, -3, 0, 0, 0, 0],
                    [0, 0, 0, 0, -3, 1, 1, 1],
                    [0, 0, 0, 0, 1, -2, 0, 1],
                    [0, 0, 0, 0, 1, 0, -1, 0],
                    [0, 0, 0, 0, 1, 1, 0, -2]])


def plot_stuff(x, graph):
    plt.clf()
    plt.ylim((1,0))
    plt.xlim((1,0))
    plt.scatter(x[:,0], x[:,1])
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if i != j and graph[i][j] == 1:
                plt.plot([x[i,0], x[j,0]], [x[i,1], x[j,1]], color='k', linestyle='-', linewidth=1)
    plt.show(block=False)
    plt.pause(.1)
    # input('press enter...')

def make_consensus(graph):
    x = np.random.random(size=graph.shape[0]*2).reshape([-1, 2])

    plot_stuff(x, graph)

    eigenvalues = np.round(np.linalg.eig(-graph)[0], decimals = 6)
    normalizer = np.max(eigenvalues)  # 30.  # graph.shape[0]
    #print(np.average(x, axis=0))

    for i in itertools.count():
        x = x + np.matmul(graph, x) / normalizer
        plot_stuff(x, graph)
        # print(str(i + 1) + ':\t' + str(x))
        if np.average(np.abs((x + np.matmul(graph, x) / normalizer) - x)) < 0.0001:
            break
    return x

make_consensus(graph_1)
make_consensus(graph_2)
make_consensus(graph_3)
make_consensus(graph_4)
make_consensus(graph_5)

print('\neigenvalues:')
print('graph_1')
print(np.round(np.sort(np.linalg.eig(-graph_1)[0]), decimals=6))
print('\ngraph_2')
print(np.round(np.sort(np.linalg.eig(-graph_2)[0]), decimals=6))
print('\ngraph_3')
print(np.round(np.sort(np.linalg.eig(-graph_3)[0]), decimals=6))
print('\ngraph_4')
print(np.round(np.sort(np.linalg.eig(-graph_4)[0]), decimals=6))
print('\ngraph_5')
print(np.round(np.sort(np.linalg.eig(-graph_5)[0]), decimals=6))
