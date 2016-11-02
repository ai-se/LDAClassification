from __future__ import print_function, division
from collections import Counter
from pdb import set_trace
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.feature_extraction import FeatureHasher
from sklearn import naive_bayes
from sklearn import tree
import random
from random import randint
from time import time
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import *

def iqr(arr):
    return np.percentile(arr,75)-np.percentile(arr,25)


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

def ig(sub,ind,ind2):

    def nlgn(n):
        if n==0:
            return 0
        else:
            return n*np.log2(n)

    a,b=docfre(sub,ind,ind2)

    keys=list(set(a.keys()+b.keys()))

    num_a=len(ind)
    num_b=len(ind2)
    score={}
    score2={}
    for key in keys:
        if key not in a.keys():
            a[key]=0
        if key not in b.keys():
            b[key]=0
        score[key]=float(a[key])/(a[key]+b[key])
        score2[key]=float(num_a-a[key])/(num_a-a[key]+num_b-b[key])

    # score[key] is P(pos|key)
    # score2[key] is P(pos|key-)

    Ppos=num_a/(num_a+num_b)
    G={}
    for key in keys:
        G[key]=-nlgn(Ppos)+nlgn(score[key])+nlgn(score2[key])


    return G

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

"make feature matrix and also return vocabulary"
def make_feature_voc(corpus,sel="tfidf",fea='tf',norm="l2row",n_features=10000):

    keys=np.array([])
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

    return matt,keys

"Preprocessing: stemming + stopwords removing"
def process(txt):
  stemmer = PorterStemmer()
  cachedStopWords = stopwords.words("english")
  return ' '.join([stemmer.stem(word) for word \
                   in txt.lower().split() if word not \
                   in cachedStopWords and len(word)>1])

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
                    dict.append(Counter(process(row).split()))
                elif pre=="bigram":
                    tm=process(row).split()
                    temp=[tm[i]+' '+tm[i+1] for i in xrange(len(tm)-1)]
                    dict.append(Counter(temp+tm))
                elif pre=="trigram":
                    tm=process(row).split()
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