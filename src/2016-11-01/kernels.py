from __future__ import print_function, division
from collections import Counter
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import time
from scipy.sparse import csr_matrix

from sklearn.feature_extraction import FeatureHasher
from random import randint, random, seed, shuffle
from sk import rdivDemo
import pickle
from demos import *
from ABCD import ABCD


## This piece of code is taken from Zhe
"term frequency "
def token_freqs(doc):
    return Counter(doc[1:])


"tf"
def tf(corpus):
    mat=[token_freqs(doc) for doc in corpus]
    return mat

"feature tfidf"
def tfidf_fea(corpus):
    word={}
    doc={}
    docs=0
    for row_c in corpus:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    for row in corpus:
        for key in row:
            #row[key]=(1+np.log(row[key]))*np.log(docs/doc[key])
            row[key]=row[key]*np.log(docs/doc[key])
    return corpus



"tf-idf"
def tf_idf(corpus):
    word={}
    doc={}
    docs=0
    for row_c in corpus:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    tfidf={}
    words=sum(word.values())
    for key in doc.keys():
        tfidf[key]=word[key]/words*np.log(docs/doc[key])
    return tfidf

"tf-idf_incremental"
def tf_idf_inc(row,word,doc,docs):
    docs+=1
    for key in row.keys():
        try:
            word[key]+=row[key]
        except:
            word[key]=row[key]
        try:
            doc[key]+=1
        except:
            doc[key]=1

    return word,doc,docs


"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat


"hashing trick"
def hash(mat,n_features=100, non_negative=True):
    if type(mat[0])==type('str') or type(mat[0])==type(u'unicode'):
        hasher = FeatureHasher(n_features=n_features, input_type='string', non_negative=non_negative)
    else:
        hasher = FeatureHasher(n_features=n_features, non_negative=non_negative)
    X = hasher.transform(mat)
    return X

def docfre(sub,ind,ind2):
    word={}
    doc={}
    docs=0
    for row_c in sub[ind]:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    a=doc
    word={}
    doc={}
    docs=0
    for row_c in sub[ind2]:
        word,doc,docs=tf_idf_inc(row_c,word,doc,docs)
    b=doc
    return a,b

"make feature matrix"
def make_feature(corpus,sel="tfidf",fea='tf',norm="l2row",n_features=10000):

    if sel=="hash":
        if fea=='tfidf_fea':
            corpus=tfidf_fea(corpus)
        matt=hash(corpus,n_features=n_features,non_negative=True)
        corpus=[]
        if norm=="l2row":
            matt=l2normalize(matt)
        elif norm=="l2col":
            matt=l2normalize(matt.transpose()).transpose()

    else:
        score={}
        if sel=="tfidf":
            score=tf_idf(corpus)
        elif sel=="docfre":
            word={}
            docs=0
            for row_c in corpus:
                word,score,docs=tf_idf_inc(row_c,word,score,docs)


        keys=np.array(score.keys())[np.argsort(score.values())][-n_features:]

        if fea=='tfidf_fea':
            corpus=tfidf_fea(corpus)
        data=[]
        r=[]
        col=[]
        num=len(corpus)
        for i,row in enumerate(corpus):
            tmp=0
            for key in keys:
                if key in row.keys():
                    data.append(row[key])
                    r.append(i)
                    col.append(tmp)
                tmp=tmp+1
        corpus=[]
        matt=csr_matrix((data, (r, col)), shape=(num, n_features))
        data=[]
        r=[]
        col=[]
        if norm=="l2row":
            matt=l2normalize(matt)
        elif norm=="l2col":
            matt=l2normalize(matt.transpose()).transpose()
    return matt


"split data according to target label"


def split_two(corpus, label):
    pos = []
    neg = []
    for i, lab in enumerate(label):
        if lab == 'pos':
            pos.append(i)
        else:
            neg.append(i)
    positive = corpus[pos]
    negative = corpus[neg]
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


def do_SVM(train_data, test_data, train_label, test_label,learner='',ker=''):
    clf = svm.SVC(kernel=ker)
    clf.fit(train_data, train_label)
    prediction = clf.predict(test_data)
    abcd = ABCD(before=test_label, after=prediction)
    F2 = np.array([k.stats()[-1] for k in abcd()])
    labeltwo = list(set(test_label))
    if labeltwo[0] == 'pos':
        labelone = 0
    else:
        labelone = 1
    try:
        return F2[labelone]
    except:
        pass


"cross validation"


def cross_val(pos=np.array([]),neg=np.array([]),folds=5,kernel=[]):
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
        return corpus[i_major], corpus[i_minor]

    "generate training set and testing set"

    def train_test(pos, neg, folds, index,issmote="smote",neighbors=5):
        pos_train, pos_test = cross_split(pos, folds=folds, index=index)
        neg_train, neg_test = cross_split(neg, folds=folds, index=index)
        if issmote == "smote":
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

    #print(pos)
    res={}
    for k in kernel:
        start_time = time.time()
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
                    train_test(pos, neg, folds=folds, index=index)
                "SVM"
                result.append(do_SVM(data_train, data_test, label_train, label_test, ker=k))
        res[k]=result
        print(result)
        print("\nTotal Runtime for %s in a %s-way cross val: --- %s seconds ---\n" % (k,str(folds),time.time() - start_time))
    return res




"Load data from file to list of lists"
def readfile_binary(filename='',thres=[0.02,0.07],pre='stem'):
    dict=[]
    label=[]
    targetlabel=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row=doc.lower().split(' >>> ')[0]
                label.append(doc.lower().split(' >>> ')[1].split()[0])
                if pre=='stem':
                    dict.append(Counter(row.split()))
                elif pre=="bigram":
                    tm=row.split()
                    temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    dict.append(Counter(temp+tm))
                elif pre=="trigram":
                    tm=row.split()
                    #temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    temp2=[tm[i]+' '+tm[i+1]+' '+tm[i+2] for i in xrange(len(tm)-2)]
                    dict.append(Counter(temp2+tm))
                else:
                    dict.append(Counter(row.split()))
            except:
                pass
    labellst=Counter(label)
    n=sum(labellst.values())
    while True:
        for l in labellst:
            if labellst[l]>n*thres[0] and labellst[l]<n*thres[1]:
                targetlabel=l
                break
        if targetlabel:
            break
        thres[1]=2*thres[1]
        thres[0]=0.5*thres[0]

    for i,l in enumerate(label):
        if l == targetlabel:
            label[i]='pos'
        else:
            label[i]='neg'
    label=np.array(label)
    print("Target Label: %s" %targetlabel)
    return label, dict

def _test(res=''):

    filepath = '/share/aagrawa8/Data/SE/'

    #filepath = '/Users/amrit/GITHUB/LDAClassification/dataset/SE/'

    thres = [0.02, 0.05]
    issel=['tfidf','hash']
    ker=['linear', 'poly', 'rbf', 'sigmoid']
    filenamelist=[]
    filenamelist.append(res)
    #filenamelist = ['cs', 'diy', 'academia', 'judaism', 'photo', 'rpg', 'scifi', 'ux']


    label, dict = readfile_binary(filename=filepath + res + '.txt', thres=thres)



    F_final = {}
    for filename in filenamelist:
        temp_file = {}
        for feature in issel:
            data = make_feature(dict, sel=feature, fea='tf', norm="l2row", n_features=4000)
            split = split_two(corpus=data, label=label)
            pos = split['pos'].toarray()
            neg = split['neg'].toarray()
            temp_file[feature] =cross_val(pos=pos,neg=neg, folds=5,  kernel=ker)
        F_final[filename] = temp_file

    with open('dump/'+res+'_kernels_features.pickle', 'wb') as handle:
        pickle.dump(F_final, handle)
    print(F_final)


if __name__ == '__main__':
    eval(cmd())