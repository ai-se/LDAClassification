from __future__ import print_function, division

__author__ = 'amrit'

from random import randint, random, seed, shuffle, sample
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import sys
import DE_topics
from scipy.sparse import csr_matrix

sys.dont_write_bytecode = True

from ABCD import ABCD


def main(*x,**r):
    #print(np.array(x))
    data=np.multiply(np.array(r['data']),np.array(x))
    data = csr_matrix(data)
    data = l2normalize(data).toarray()
    return _test(data=data, targetlist=r['target'],tune=r['tune'],perc=r['percentage']),0


def l2normalize(mat):
    mat = mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor = np.linalg.norm(mat[i].data, 2)
        if not nor == 0:
            for k in mat[i].indices:
                mat[i, k] = mat[i, k] / nor
    return mat

def split_two(corpus, label):
    pos = []
    neg = []
    for i, lab in enumerate(label):
        if lab == 'pos':
            pos.append(i)
        else:
            neg.append(i)
    #print(pos)
    ## corpus is of dictionary type.
    positive = corpus[pos]
    negative = corpus[neg]
    #print(positive)
    return {'pos': positive, 'neg': negative}


"smote"


def smote(data, num, k=5):
    corpus = []
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, num):
        mid = randint(0, len(data) - 1)
        nn = indices[mid, randint(1, k)]
        datamade = []
        for j in range(0, len(data[mid])):
            gap = random()
            datamade.append((data[nn, j] - data[mid, j]) * gap + data[mid, j])
        corpus.append(datamade)
    corpus = np.array(corpus)
    return corpus


"SVM"


def do_SVM(train_data, test_data, train_label, test_label):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    abcd = ABCD(before=test_label, after=prediction)
    F = np.array([k.stats()[-1] for k in abcd()])
    labeltwo = list(set(test_label))
    #print(labeltwo)
    if labeltwo[0] == 'pos':
        labelone = 0
    else:
        labelone = 1
    try:
        return F[labelone]
    except:
        pass


"cross validation"

## this is 80% training, and 20% testing
def cross_val(data=[],  folds=5, target=[],tune='yes',perc=50):
    "split for cross validation"

    def cross_split(corpus, folds, index):
        i_major = []
        i_minor = []
        l = len(corpus)
        for i in range(0, folds):
            if i == index:
                i_minor.extend(range(int(i * l / folds), int((i + 1) * l / folds)))
            else:
                i_major.extend(range(int(i * l / folds), int((i + 1) * l / folds)))
        return  corpus[i_major],corpus[i_minor]

    "generate training set and testing set"

    def tune_train_test(pos, neg, folds, index, neighbors=5):
        pos_train, pos_test = cross_split(pos, folds=5, index=index)
        neg_train, neg_test = cross_split(neg, folds=5, index=index)

        ##smoting
        num = int((len(pos_train) + len(neg_train)) / 2)
        pos_train = smote(pos_train, num, k=neighbors)
        neg_train = neg_train[np.random.choice(len(neg_train), num, replace=False)]

        data_train = np.vstack((pos_train, neg_train))
        data_test = np.vstack((pos_test, neg_test))
        label_train = ['pos'] * len(pos_train) + ['neg'] * len(neg_train)
        label_test = ['pos'] * len(pos_test) + ['neg'] * len(neg_test)

        "Shuffle"
        tmp = range(0, len(label_train))
        shuffle(tmp)
        data_train = data_train[tmp]
        label_train = np.array(label_train)[tmp]

        tmp = range(0, len(label_test))
        shuffle(tmp)
        data_test = data_test[tmp]
        label_test = np.array(label_test)[tmp]

        return data_train, data_test, label_train, label_test

    def no_train_test(data_train, train_label, neighbors=5):
        pos_train=[]
        neg_train=[]
        for j,i in enumerate(train_label):
            if i=='pos':
                pos_train.append(data_train[j])
            else:
                neg_train.append(data_train[j])
        ##smoting
        pos_train=np.array(pos_train)
        neg_train=np.array(neg_train)
        num = int((len(pos_train) + len(neg_train)) / 2)
        pos_train = smote(pos_train, num, k=neighbors)
        neg_train = neg_train[np.random.choice(len(neg_train), num, replace=False)]

        data_train1 = np.vstack((pos_train, neg_train))
        label_train = ['pos'] * len(pos_train) + ['neg'] * len(neg_train)

        return data_train1, label_train

    if tune=='on':
        split = split_two(corpus=data, label=target)
        pos = split['pos']
        neg = split['neg']

        result = []
        for i in range(folds):
            tmp = range(0, len(pos))
            shuffle(tmp)
            pos = pos[tmp]
            tmp = range(0, len(neg))
            shuffle(tmp)
            neg = neg[tmp]
            for index in range(folds):
                data_train, data_test, label_train, label_test = \
                    tune_train_test(pos, neg, folds=folds, index=index, neighbors=5)
                "SVM"
                result.append(do_SVM(data_train, data_test, label_train, label_test))
        return result
    else:
        split=DE_topics.split_two(data,target)
        cut_pos,cut_neg=DE_topics.cut_position(split['pos'],split['neg'],percentage=perc)
        data_train,label_train,data_test,label_test=DE_topics.divide_train_test(split['pos'],split['neg'],cut_pos,cut_neg)
        data_train, label_train=no_train_test(data_train,label_train,neighbors=5)
        return do_SVM(data_train, data_test, label_train, label_test)


def _test(data=[],file='', targetlist=[],tune='',perc=0):

    return np.median(cross_val(data=data,folds=1,target=targetlist,tune=tune,perc=perc))

#SE0: Counter({'no': 6008, 'yes': 309})
#SE1  Counter({'no': 47201, 'yes': 1441})
#SE3  Counter({'no': 83583, 'yes': 654})
#SE6  Counter({'no': 15865, 'yes': 439})\
#SE8  Counter({'no': 58076, 'yes': 195})

