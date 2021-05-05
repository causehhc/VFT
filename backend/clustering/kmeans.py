# from database import MySqlHelper
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MeanShift
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
    sim = np.dot(text_weight[i], text_weight[j]) / (norm(text_weight[i]) * norm(text_weight[j]))
    sim = int(100 - (sim * 100))
    return sim


def get_nodes_links(path):
    stop_words = open(path, 'r', encoding='utf-8').read().replace('\n', ' ').split()
    t1 = time.time()
    sql = MySqlHelper()
    corpus_orin = sql.info_getAll('title')
    # corpus_orin = corpus_orin[0:100]
    # corpus_orin = ['教育经历和工作经历没有空间',
    #                '我们能改进的只有专业技能和项目经历了',
    #                '我有两家大厂工作经验',
    #                '精通来描述一项专业技能的掌握程度']
    sent_words = [list(jieba.cut(sent0)) for sent0 in corpus_orin]
    corpus = [' '.join(sent0) for sent0 in sent_words]
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    text = vectorizer.fit_transform(corpus)
    text_weight = text.toarray()

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(text_weight)
    # meanshift = MeanShift()
    # meanshift.fit(text_weight)

    # print(text.toarray())

    t2 = time.time()
    nodes = []
    links = []
    tk = 0
    len_text_weight = len(text_weight)
    for i in range(len_text_weight):
        tmp_node = {'id': corpus_orin[i], 'group': str(kmeans.labels_[i])}
        # tmp_node = {'id': corpus_orin[i], 'group': str(meanshift.labels_[i])}
        nodes.append(tmp_node)
        for j in range(i + 1, len_text_weight-1):
            t22 = time.time()
            sim = get_sim(text_weight, i, j)
            t222 = time.time()
            tk += t222-t22
            tmp_edge = {'source': corpus_orin[i], 'target': corpus_orin[j], 'value': sim}
            if sim < 85:
                links.append(tmp_edge)
            elif j == i + 1:
                links.append(tmp_edge)

    t3 = time.time()

    print(t2-t1)
    print(t3-t2, tk)
    print(t3-t1)
    return nodes, links


def main():
    # main_engine()
    nodes, links = get_nodes_links('./stopwords-master/baidu_stopwords.txt')
    print(nodes)
    print(links)


if __name__ == '__main__':
    main()
