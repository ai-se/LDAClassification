from __future__ import print_function, division

__author__ = 'amrit'

from demos import atom
from demos import cmd
import collections
from topicsgibbs import *
import random
import time
import copy
import operator
import os, pickle
import svmtopics
import sys
import numpy as np

sys.dont_write_bytecode = True

__all__ = ['DE']
Individual = collections.namedtuple('Individual', 'ind fit1 fit2')


class DE(object):
    def __init__(self, x='rand', y=1, z='bin', F=0.3, CR=0.7):
        self.x = x
        self.y = y
        self.z = z
        self.F = F
        self.CR = CR

    # TODO: add a fitness_aim param?
    # TODO: add a generic way to generate initial pop?
    def solve(self, fitness, initial_population, iterations=10, **r):
        current_generation=[]
        for ind in initial_population:
            a,b=fitness(*ind, **r)
            current_generation.append(Individual(ind, a,b))

        l=[]
        for i in current_generation:
            l.append([i.ind,i.fit1,i.fit2])
        for _ in range(iterations):
            trial_generation = []

            for ind in current_generation:
                v = self._extrapolate(ind,current_generation)
                a1,b1=fitness(*v, **r)
                trial_generation.append(Individual(v, a1,b1))
                l.append([v, a1,b1])

            current_generation = self._selection(current_generation,
                                                 trial_generation)

        best_index = self._get_best_index(current_generation)
        return current_generation[best_index].ind, [current_generation[best_index].fit1,current_generation[best_index].fit2], l

    def select3others(self,population):
        popu=copy.deepcopy(population)
        x= random.randint(0, len(popu)-1)
        x1=popu[x]
        popu.pop(x)
        y= random.randint(0, len(popu)-1)
        y1=popu[y]
        popu.pop(y)
        z= random.randint(0, len(popu)-1)
        z1=popu[z]
        popu.pop(z)
        return x1.ind,y1.ind,z1.ind

    def _extrapolate(self, ind, population):
        if (random.random() < self.CR):
            x,y,z=self.select3others(population)
            #print(x,y,z)
            mutated=[x[0] + self.F*(y[0] - z[0]), x[1] + self.F*(y[1] - z[1]), x[2] + self.F*(y[2] - z[2])]

            check_mutated= [max(bounds[0][0], min(mutated[0], bounds[0][1])),max(bounds[1][0], min(mutated[1], bounds[1][1])) ,max(bounds[2][0], min(mutated[2], bounds[2][1]))]
            return check_mutated
        else:
            return ind.ind

    def _selection(self, current_generation, trial_generation):
        generation = []

        for a, b in zip(current_generation, trial_generation):
            if (a.fit1+np.median(a.fit2)) >= (b.fit1+np.median(b.fit2)):
                generation.append(a)
            else:
                generation.append(b)

        return generation

    def _get_indices(self, n, upto, but=None):
        candidates = list(range(upto))

        if but is not None:
            # yeah O(n) but random.sample cannot use a set
            candidates.remove(but)

        return random.sample(candidates, n)

    def _get_best_index(self, population):
        global max_fitness
        best = 0

        for i, x in enumerate(population):
            if (x.fit1+np.median(x.fit2)) >= max_fitness:
                best = i
                max_fitness = x.fit1+np.median(x.fit2)
        return best

    def _set_x(self, x):
        if x not in ['rand', 'best']:
            raise ValueError("x should be either 'rand' or 'best'.")

        self._x = x

    def _set_y(self, y):
        if y < 1:
            raise ValueError('y should be > 0.')

        self._y = y

    def _set_z(self, z):
        if z != 'bin':
            raise ValueError("z should be 'bin'.")

        self._z = z

    def _set_F(self, F):
        if not 0 <= F <= 2:
            raise ValueError('F should belong to [0, 2].')

        self._F = F

    def _set_CR(self, CR):
        if not 0 <= CR <= 1:
            raise ValueError('CR should belong to [0, 1].')

        self._CR = CR

    x = property(lambda self: self._x, _set_x, doc='How to choose the vector '
                                                   'to be mutated.')
    y = property(lambda self: self._y, _set_y, doc='The number of difference '
                                                   'vectors used.')
    z = property(lambda self: self._z, _set_z, doc='Crossover scheme.')
    F = property(lambda self: self._F, _set_F, doc='Weight used during '
                                                   'mutation.')
    CR = property(lambda self: self._CR, _set_CR, doc='Weight used during '
                                                      'bin crossover.')


def cmd(com="demo('-h')"):
    "Convert command line to a function call."
    if len(sys.argv) < 2: return com

    def strp(x): return isinstance(x, basestring)

    def wrap(x): return "'%s'" % x if strp(x) else str(x)

    words = map(wrap, map(atom, sys.argv[2:]))
    return sys.argv[1] + '(' + ','.join(words) + ')'

"Load data from file to list of lists"
def readfile1(filename='',thres=[0.02,0.07]):
    dict=[]
    label=[]
    targetlabel=[]
    with open(filename,'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().split(' >>> ')[0].strip()
                label.append(doc.lower().split(' >>> ')[1].split()[0])
                dict.append(row)
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
    return dict, label

def split_two(corpus=[], label=np.array([])):
    pos = []
    neg = []
    for i, lab in enumerate(label):
        if lab == 'pos':
            pos.append(corpus[i])
        else:
            neg.append(corpus[i])

    return {'pos': pos, 'neg': neg}

def _topics(res=''):
    #fileB = ['pitsA', 'pitsB', 'pitsC', 'pitsD', 'pitsE', 'pitsF', 'processed_citemap.txt']
    #fileB = ['SE0', 'SE6', 'SE1', 'SE8', 'SE3']
    filepath = '/share/aagrawa8/Data/SE/'
    start_time = time.time()
    #filepath='/Users/amrit/GITHUB/LDAClassification/dataset/SE/'

    random.seed(1)
    global bounds
    cross_tune='no'
    grow_oracle='yes'
    data_samples, labellist = readfile1(filepath + str(res)+'.txt')
    split = split_two(corpus=data_samples, label=labellist)
    pos = np.array(split['pos'])
    neg = np.array(split['neg'])
    cut_pos = int(len(pos) * 80 / 100)
    cut_neg = int(len(neg) * 80 / 100)
    ##list of f2 scores
    lis=[]
    #dictionary containing bestone, time for 1 run, f2
    cross={}
    #dictionary containing cross, lis,full time
    file={}
    for folds in range(5):
        start_time1 = time.time()
        pos_shuffle = range(0, len(pos))
        neg_shuffle=range(0, len(neg))
        shuffle(pos_shuffle)
        shuffle(neg_shuffle)
        pos=pos[pos_shuffle]
        neg=neg[neg_shuffle]

        data_train, train_label=list(pos)[:cut_pos]+list(neg)[:cut_neg],['pos']*cut_pos + ['neg']*cut_neg
        data_test, test_label = list(pos)[cut_pos:]+list(neg)[cut_neg:], ['pos']*(len(pos)-cut_pos) + ['neg']*(len(neg)-cut_neg)

        # stability score format dict, file,lab=score
        # parameter variations (k,a,b), format, list of lists, file,lab=[[k,a,b], Rn score, fscore]
        #final_para_dic={}
        # final paras and scores, file, lab=[[k,a,b],[r, f1]]
        de = DE(F=0.7, CR=0.3, x='rand')

        global max_fitness
        max_fitness = 0
        pop = [[random.randint(bounds[0][0], bounds[0][1]), random.uniform(bounds[1][0], bounds[1][1]),
                        random.uniform(bounds[2][0], bounds[2][1])]
                       for _ in range(10)]
        v, score, final_para_dic = de.solve(main, pop, iterations=2, file=res, term=7, data_samples=data_train,target=train_label,tune='yes')
        ##score is a list of [jaccard and fscore]
        bestone = [v,score]
        # runtime,format dict, file,=runtime in secs
        l=bestone
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(data_train+data_test)
        lda1 = lda.LDA(n_topics=int(l[0][0]), alpha=l[0][1], eta=l[0][2], n_iter=200)
        lda1.fit_transform(tf)
        tops = lda1.doc_topic_
        f2=svmtopics.main(data=tops, file=res, target=train_label+test_label,tune='no')
        lis.append(f2)
        time2 = time.time() - start_time1
        cross[folds] = [ bestone,time2,f2]
        print([bestone,time2,f2])
        print("\nRuntime for 1 loop of DE termination: --- %s seconds ---\n" % (time2))
    time1=time.time() - start_time
    file[res]=[cross,lis,time1]
    print("\nTotal Runtime: --- %s seconds ---\n" % (time.time() - start_time))
    with open('dump/DE_log_tune_grow_oracle'+res+'.pickle', 'wb') as handle:
        pickle.dump(file, handle)


    ##untuned experiment

    '''l={}
    temp1={}
    ##with term = 7, 10
    for i in [7,10]:
        temp = {}
        for j in [10,20,40,80,200]:
            temp[j]=main(j,0.1,0.01,file=res, term=i, data_samples=data_samples,target=labellist)
        temp1[i]=temp
    l[res]=temp1
    with open('dump/untuned_class_topics_'+res+'.pickle', 'wb') as handle:
        pickle.dump(l, handle)
    print("\nTotal Runtime: --- %s seconds ---\n" % (time.time() - start_time))'''


bounds = [(70, 150), (0.1, 1), (0.01, 1)]
max_fitness = 0
if __name__ == '__main__':
    eval(cmd())
