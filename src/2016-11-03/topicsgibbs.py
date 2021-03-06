from __future__ import print_function, division

__author__ = 'amrit'

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
                    j_score.append(sum / float(4))
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
        for i in range(1):
            #data_samples = readfile1(filepath + str(file1))

            # shuffling the list
            shuffle(x)
            data=[data[k] for k in x]
            tar=[tar[k] for k in x]

            tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tf = tf_vectorizer.fit_transform(data)

            lda1 = lda.LDA(n_topics=int(l[0]), alpha=l[1], eta=l[2],n_iter=50)

            lda1.fit_transform(tf)
            tops = lda1.doc_topic_
            topic_word = lda1.topic_word_
            log=1 / (-lda1.loglikelihood()) * 100000
            tf_feature_names = tf_vectorizer.get_feature_names()
            topics.extend(get_top_words(lda1, path1, tf_feature_names, n_top_words, i=i, file1=file1))
    return topics,tops,topic_word,tf_feature_names,tar,log


def main(*x, **r):
    # 1st r
    start_time = time.time()
    base = '/share/aagrawa8/Data/SE/'
    #base = '/home/amrit/GITHUB/LDAClassification/results/SE/'
    path = os.path.join(base, 'jaccard_tune_grow_oracle', r['file'], str(r['term']))
    #path = os.path.join(base, 'untuned_svm_topics_smote', r['file'], str(r['term']))
    if not os.path.exists(path):
        os.makedirs(path)
    l = np.asarray(x)
    b = int(l[0])
    path1 = path + "/K_" + str(b) + "_a_" + str(l[1]) + "_b_" + str(l[2]) + ".txt"
    with open(path1, "w") as f:
        f.truncate()

    topics,tops,word,corpus,tar,log = _test_LDA(l, path1, file=r['file'],data_samples=r['data_samples'],target=r['target'])

    top=[]
    fscore = svmtopics.main(data=tops, file=r['file'], target=tar,tune=r['tune'])
    for i in topics:
        temp=str(i.encode('ascii','ignore'))
        top.append(temp)
    a = jaccard(b, score_topics=top, term=r['term'])
    fo = open(path1, 'a+')
    #fo.write("\nScore: " + str(a))
    fo.write("\nScore: " + str(a))
    fo.write("\nRuntime: --- %s seconds ---\n" % (time.time() - start_time))
    fo.close()

    return a,fscore
