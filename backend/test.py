# ----------------------------------------------
# Project： calculate diameter of graph
# Using floyd algorithm
# ----------------------------------------------


# define function: print shortest path
def getPath(i, j):
    if i != j:
        if path[i][j] == -1:
            print('-', j + 1, end='')
        else:
            getPath(i, path[i][j])
            getPath(path[i][j], j)


def printPath(i, j):
    print(' Path:', i + 1, end='')
    getPath(i, j)
    print()


if __name__ == '__main__':
    print('---------------- Program start ----------------')
    # read data
    # flag = input('please input type of graph(1:directed graph; 2:undirected graph): ')
    flag = 2
    # vertex, edge = input('please input the number of vertex and edge: ').strip().split()
    vertex = len(nodes)
    edge = len(links)

    # initialized
    flag = int(flag)
    vertex = int(vertex)
    edge = int(edge)
    inf = 99999999
    dis = []  # matrix of the shortest distance
    path = []  # record the shortest path
    for i in range(vertex):
        dis += [[]]
        for j in range(vertex):
            if i == j:
                dis[i].append(0)
            else:
                dis[i].append(inf)
    for i in range(vertex):
        path += [[]]
        for j in range(vertex):
            path[i].append(-1)

    # read weight information
    print('please input weight info(v1 v2 w[v1,v2]): ')
    for i in range(edge):
        u, v, w = input().strip().split()
        u, v, w = int(u) - 1, int(v) - 1, int(w)
        if flag == 1:
            dis[u][v] = w
        elif flag == 2:
            dis[u][v] = w
            dis[v][u] = w
    print('the weight matrix is:')
    for i in range(vertex):
        for j in range(vertex):
            if dis[i][j] != inf:
                print('%5d' % dis[i][j], end='')
            else:
                print('%5s' % '∞', end='')
        print()

    # floyd algorithm
    for k in range(vertex):
        for i in range(vertex):
            for j in range(vertex):
                if dis[i][j] > dis[i][k] + dis[k][j]:
                    dis[i][j] = dis[i][k] + dis[k][j]
                    path[i][j] = k
    print('===========================================')

    # output the result
    print('output the result:')
    if flag == 1:
        for i in range(vertex):
            for j in range(vertex):
                if (i != j) and (dis[i][j] != inf):
                    print('v%d ----> v%d  tol_weight:'
                          '%3d' % (i + 1, j + 1, dis[i][j]))
                    printPath(i, j)
                if (i != j) and (dis[i][j] == inf):
                    print('v%d ----> v%d  tol_weight:'
                          '  ∞' % (i + 1, j + 1))
                    printPath(i, j)

    if flag == 2:
        for i in range(vertex):
            for j in range(i + 1, vertex):
                print('v%d <----> v%d  tol_weight:'
                      '%3d' % (i + 1, j + 1, dis[i][j]), '', end='')
                printPath(i, j)
    print()
    for i in range(vertex):
        for j in range(vertex):
            if dis[i][j] == inf:
                dis[i][j] = 0
    # max(max(dis)): the max item of two dimension matrix
    print('>> the diameter of graph: %d <<' % max(max(dis)))
    print('-------------- Program end ----------------')
