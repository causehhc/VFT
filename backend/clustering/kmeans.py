# from database import MySqlHelper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import jieba
from scipy.linalg import norm
import numpy as np
from backend.database import MySqlHelper


def main_engine():
    """
            1、加载语料
        """
    print('1、加载语料')
    sql = MySqlHelper()
    corpus = sql.info_getAll('title')
    sent_words = [list(jieba.cut(sent0)) for sent0 in corpus]
    corpus = [' '.join(sent0) for sent0 in sent_words]

    '''
        2、计算tf-idf设为权重
    '''
    print('2、计算tf-idf设为权重')
    stop_words = open('./stopwords-master/baidu_stopwords.txt', 'r', encoding='utf-8').read().split('\n')
    print(stop_words)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    text = vectorizer.fit_transform(corpus)
    text_weight = text.toarray()
    print(len(vectorizer.get_feature_names()))
    print(vectorizer.get_feature_names())
    print(text_weight)
    '''
        3、对向量进行聚类
    '''
    print('3、对向量进行聚类')
    # 指定分成7个类
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(text_weight)

    # 打印出各个族的中心点
    print(kmeans.cluster_centers_)
    for index, label in enumerate(kmeans.labels_, 1):
        print("index: {}, label: {}".format(index, label))

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(kmeans.inertia_))

    '''
        4、可视化
    '''
    print('4、可视化')
    # 使用T-SNE算法，对权重进行降维，准确度比PCA算法高，但是耗时长
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(text_weight)

    x = []
    y = []

    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    plt.scatter(x, y, c=kmeans.labels_, marker="x")
    plt.xticks(())
    plt.yticks(())
    plt.show()
    # plt.savefig('./sample.png', aspect=1)


def get_sim(text_weight, i, j):
    sim = np.dot(text_weight[i], text_weight[j]) / (
            norm(text_weight[i]) * norm(text_weight[j]))
    sim = int(100 - (sim * 100))
    return sim


# vertex = 0
# edge = 0
# inf = 99999999
# dis = []  # matrix of the shortest distance
# path = []  # record the shortest path


def getPath(ppp, path, i, j):
    if i != j:
        if path[i][j] == -1:
            # print('-', j, end='')
            ppp.append(j)
        else:
            getPath(ppp, path, i, path[i][j])
            getPath(ppp, path, path[i][j], j)


def printPath(path, i, j):
    ppp = [i]
    # print(' Path:', i, end='')
    getPath(ppp, path, i, j)
    # print()
    # print(ppp)
    return ppp


def get_nodes_links(path):
    sql = MySqlHelper()
    corpus_orin = sql.info_getAll('title')
    # corpus_orin = ['教育经历和工作经历没有空间',
    #                '我们能改进的只有专业技能和项目经历了',
    #                '我有两家大厂工作经验',
    #                '精通来描述一项专业技能的掌握程度']
    corpus_orin = corpus_orin[0:100]
    sent_words = [list(jieba.cut(sent0)) for sent0 in corpus_orin]
    corpus = [' '.join(sent0) for sent0 in sent_words]
    stop_words = open(path, 'r', encoding='utf-8').read().split('\n')
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    text = vectorizer.fit_transform(corpus)
    text_weight = text.toarray()

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(text_weight)

    nodes = []
    links = []
    # 打印出各个族的中心点
    # print(text_weight)
    # print(np.dot(text_weight[0], text_weight[1]) / (norm(text_weight[0]) * norm(text_weight[1])))
    # print(kmeans.cluster_centers_)
    for index, label in enumerate(kmeans.labels_, 1):
        tmp = {'id': '{}'.format(corpus_orin[index - 1]), 'group': '{}'.format(label)}
        nodes.append(tmp)

    # sim_edge_list = []
    # sim_graph = [[0 for _ in range(len(text_weight))] for _ in range(len(text_weight))]
    len_text_weight = len(text_weight)
    for i in range(len_text_weight):
        if i != len_text_weight - 1:
            tmp = {
                'source': '{}'.format(corpus_orin[i]),
                'target': '{}'.format(corpus_orin[i+1]),
                'value': '{}'.format(1),
                # 'source': i,
                # 'target': j,
                # 'value': sim
            }
            links.append(tmp)
            for j in range(i + 1, len_text_weight):
                sim = get_sim(text_weight, i, j)
                if sim < 85:
                    tmp = {
                        'source': '{}'.format(corpus_orin[i]),
                        'target': '{}'.format(corpus_orin[j]),
                        'value': '{}'.format(sim),
                        # 'source': i,
                        # 'target': j,
                        # 'value': sim
                    }
                    # print(tmp)
                    # sim_edge_list.append(tmp)
                    # sim_graph[i][j] = sim
                    links.append(tmp)
        # print(sim_graph)
    print(len(links))
    # # ===================================================================================
    # # initialized
    # vertex = len_text_weight
    # edge = len(sim_edge_list)
    # inf = 99999999
    # dis = []  # matrix of the shortest distance
    # path = []  # record the shortest path
    # for i in range(vertex):
    #     dis += [[]]
    #     for j in range(vertex):
    #         if i == j:
    #             dis[i].append(0)
    #         else:
    #             dis[i].append(inf)
    # for i in range(vertex):
    #     path += [[]]
    #     for j in range(vertex):
    #         path[i].append(-1)
    # # read weight information
    # print('please input weight info(v1 v2 w[v1,v2]): ')
    # for i in range(edge):
    #     # u, v, w = input().strip().split()
    #     u = sim_edge_list[i]['source']
    #     v = sim_edge_list[i]['target']
    #     w = sim_edge_list[i]['value']
    #     u, v, w = u, v, w
    #     print(u, v, w)
    #     dis[u][v] = w
    #     dis[v][u] = w
    # print('the weight matrix is:')
    # for i in range(vertex):
    #     for j in range(vertex):
    #         if dis[i][j] != inf:
    #             print('%5d' % dis[i][j], end='')
    #         else:
    #             print('%5s' % '∞', end='')
    #     print()
    # # floyd algorithm
    # for k in range(vertex):
    #     for i in range(vertex):
    #         for j in range(vertex):
    #             if dis[i][j] > dis[i][k] + dis[k][j]:
    #                 dis[i][j] = dis[i][k] + dis[k][j]
    #                 path[i][j] = k
    # print('===========================================')
    # ans = []
    # # output the result
    # print('output the result:')
    # for i in range(vertex):
    #     for j in range(i + 1, vertex):
    #         # print('v%d <----> v%d  tol_weight:%3d' % (i, j, dis[i][j]), '', end='')
    #         ppp = printPath(path, i, j)
    #         if dis[i][j] != inf:
    #             for ii in range(len(ppp)):
    #                 if ii != len(ppp)-1:
    #                     if [ppp[ii], ppp[ii+1]] not in ans and [ppp[ii+1], ppp[ii]] not in ans:
    #                         print([ppp[ii], ppp[ii+1]])
    #                         ans.append([ppp[ii], ppp[ii+1]])
    # print(edge)
    # print('ppppp', len(ans))
    #
    # print()
    # for i in range(vertex):
    #     for j in range(vertex):
    #         if dis[i][j] == inf:
    #             dis[i][j] = 0
    # # max(max(dis)): the max item of two dimension matrix
    # print('>> the diameter of graph: %d <<' % max(max(dis)))
    # print('-------------- Program end ----------------')
    # # ===================================================================================

    # sim_edge_list = sorted(sim_edge_list, reverse=True, key=lambda x: x['value'])

    # for item in sim_edge_list:
    #     if item['value'] > '0.5':
    #         links.append(item)

    # hashset = set()
    # for item in sim_edge_list:
    #     if item['target'] not in hashset:
    #         hashset.add(item['source'])
    #         hashset.add(item['target'])
    #         links.append(item)
    #     elif item['source'] not in hashset:
    #         hashset.add(item['source'])
    #         links.append(item)

    return nodes, links


def main():
    # main_engine()
    nodes, links = get_nodes_links('./stopwords-master/baidu_stopwords.txt')
    # print(nodes)
    # print(links)


if __name__ == '__main__':
    main()
