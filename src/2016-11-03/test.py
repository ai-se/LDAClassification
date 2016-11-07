from __future__ import print_function, division

__author__ = 'amrit'

import numpy as np
import lda
import lda.datasets
from random import randint, random, seed, shuffle, sample
import numpy as np
import os
import lda
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import copy
import time
import svmtopics
import sys
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_samples = 2000
sys.dont_write_bytecode = True


def calculate(topics=[], lis=[], count1=0):
    count = 0
    for i in topics:
        if i in lis:
            count += 1
    if count >= count1:
        return count
    else:
        return 0


def recursion(topic=[], index=0, count1=0):
    count = 0
    global data
    # print(data)
    # print(topics)
    d = copy.deepcopy(data)
    d.pop(index)
    for l, m in enumerate(d):
        # print(m)
        for x, y in enumerate(m):
            if calculate(topics=topic, lis=y, count1=count1) != 0:
                count += 1
                break
                # data[index+l+1].pop(x)
    return count


data = []


def jaccard(a, score_topics=[], term=0):
    labels = []  # ,6,7,8,9]
    labels.append(term)
    global data
    l = []
    data = []
    file_data = {}
    for doc in score_topics:
        l.append(doc.split())
    for i in range(0, len(l), int(a)):
        l1 = []
        for j in range(int(a)):
            l1.append(l[i + j])
        data.append(l1)
    dic = {}
    for x in labels:
        j_score = []
        for i, j in enumerate(data):
            for l, m in enumerate(j):
                sum = recursion(topic=m, index=i, count1=x)
                if sum != 0:
                    j_score.append(sum / float(9))
                '''for m,n in enumerate(l):
                    if n in j[]'''
        dic[x] = j_score
        if len(dic[x]) == 0:
            dic[x] = [0]
    file_data['citemap'] = dic


# print(file_data)
    X = range(len(labels))
    Y_median = []
    Y_iqr = []
    for feature in labels:
        Y = file_data['citemap'][feature]
        Y=sorted(Y)
        return Y[int(len(Y)/2)]


def get_top_words(model, path1, feature_names, n_top_words, i=0, file1=''):
    topics = []
    fo = open(path1,'a+')
    fo.write("Run: " + str(i) + "\n")
    for topic_idx, topic in enumerate(model.components_):
        str1 = ''
        fo.write("Topic " + str(topic_idx) + ": ")
        for j in topic.argsort()[:-n_top_words - 1:-1]:
            str1 += feature_names[j] + " "
            fo.write(feature_names[j] + " ")
        topics.append(str1)
        fo.write("\n")
    fo.close()
    return topics

def readfile1(filename=''):
    dict = []
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().strip()
                dict.append(row)
            except:
                pass
    return dict


def _test_LDA(l, path1, file='',data_samples=[],target=[]):
    n_topics = 10
    n_top_words = 10

    fileB = []
    fileB.append(file)
    #filepath = '/home/amrit/GITHUB/Pits_lda/dataset/'
    topics=[]
    data=data_samples
    tar=target
    log=0

    x=list(xrange(len(data_samples)))
    for j, file1 in enumerate(fileB):
        for i in range(10):
            #data_samples = readfile1(filepath + str(file1))

            # shuffling the list
            shuffle(x)
            data=[data[k] for k in x]
            #tar=[tar[k] for k in x]

            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tf = tf_vectorizer.fit_transform(data)

            lda1 = lda.LDA(n_topics=int(l[0]), alpha=l[1], eta=l[2],n_iter=50)

            lda1.fit_transform(tf)
            tops = lda1.doc_topic_
            topic_word = lda1.topic_word_
            #log=1 / (-lda1.loglikelihood()) * 100000
            tf_feature_names = tf_vectorizer.get_feature_names()
            topics.extend(get_top_words(lda1, path1, tf_feature_names, n_top_words, i=i, file1=file1))
    return topics,tops,topic_word,tf_feature_names,tar,log


dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
topics,tops,word,corpus,tar,log = _test_LDA([200,0.1,0.01], 'test.txt', file='test.txt',data_samples=data_samples)

top=[]
for i in topics:
    temp=str(i.encode('ascii','ignore'))
    top.append(temp)
start_time=time.time()
a = jaccard(200, score_topics=top, term=7)
print(time.time()-start_time)
print(a)

'''X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
#print(titles)
model = lda.LDA(n_topics=50, alpha=0.1, eta=0.01, n_iter=75, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_  # model.components_ also works
n_top_words = 8
print(1/(-model.loglikelihood())*100000)
doc_topic = model.doc_topic_
print(doc_topic[0])
for i in range(10):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))
#0.149482999613

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_samples = 2000
n_features = 1000
n_topics = 100
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:n_samples]
print("done in %0.3fs." % (time() - t0))
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data_samples)
print("done in %0.3fs." % (time() - t0))

lda = LatentDirichletAllocation(n_topics=n_topics,doc_topic_prior=0.1, topic_word_prior=0.01, max_iter=n_topics,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)
print(1/(-lda.score(tf))*1000000)
print("done in %0.3fs." % (time() - t0))'''