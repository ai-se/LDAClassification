from __future__ import print_function, division

__author__ = 'amrit'

from demos import *
import collections
from collections import Counter
from topicsgibbs import *
import random
import time
import copy
import pickle
import svmtopics
import sys
import numpy as np
from scipy.sparse import csr_matrix

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
        current_generation = []
        for ind in initial_population:
            a, b = fitness(*ind, **r)
            current_generation.append(Individual(ind, a, b))

        l = []
        for i in current_generation:
            l.append([i.ind, i.fit1, i.fit2])
        for _ in range(iterations):
            trial_generation = []

            for ind in current_generation:
                v = self._extrapolate(ind, current_generation, bounds=r['bounds'])
                a1, b1 = fitness(*v, **r)
                trial_generation.append(Individual(v, a1, b1))
                l.append([v, a1, b1])

            current_generation = self._selection(current_generation,
                                                 trial_generation)

        best_index = self._get_best_index(current_generation)
        return current_generation[best_index].ind, [current_generation[best_index].fit1,
                                                    current_generation[best_index].fit2], l

    def select3others(self, population):
        popu = copy.deepcopy(population)
        x = random.randint(0, len(popu) - 1)
        x1 = popu[x]
        popu.pop(x)
        y = random.randint(0, len(popu) - 1)
        y1 = popu[y]
        popu.pop(y)
        z = random.randint(0, len(popu) - 1)
        z1 = popu[z]
        popu.pop(z)
        return x1.ind, y1.ind, z1.ind

    def _extrapolate(self, ind, population, bounds=[]):
        if (random.random() < self.CR):
            x, y, z = self.select3others(population)
            # print(x,y,z)
            mutated = []
            for i in range(len(x)):
                mutated.append(x[i] + self.F * (y[i] - z[i]))
            check_mutated = []
            for i in range(len(mutated)):
                check_mutated.append(max(bounds[i][0], min(mutated[i], bounds[i][1])))
            return check_mutated
        else:
            return ind.ind

    def _selection(self, current_generation, trial_generation):
        generation = []

        for a, b in zip(current_generation, trial_generation):
            if (a.fit1 + np.median(a.fit2)) >= (b.fit1 + np.median(b.fit2)):
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
            if (x.fit1 + np.median(x.fit2)) >= max_fitness:
                best = i
                max_fitness = x.fit1 + np.median(x.fit2)
        return best


"Load data from file to list of lists"


def readfile1(filename='', thres=[0.02, 0.07]):
    dict = []
    label = []
    targetlabel = []
    with open(filename, 'r') as f:
        for doc in f.readlines():
            try:
                row = doc.lower().split(' >>> ')[0].strip()
                label.append(doc.lower().split(' >>> ')[1].split()[0])
                dict.append(row)
            except:
                pass
    labellst = Counter(label)
    n = sum(labellst.values())
    while True:
        for l in labellst:
            if labellst[l] > n * thres[0] and labellst[l] < n * thres[1]:
                targetlabel = l
                break
        if targetlabel:
            break
        thres[1] = 2 * thres[1]
        thres[0] = 0.5 * thres[0]

    for i, l in enumerate(label):
        if l == targetlabel:
            label[i] = 'pos'
        else:
            label[i] = 'neg'
    label = np.array(label)
    print("Target Label: %s" % targetlabel)
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


def cut_position(pos, neg, percentage=0):
    return int(len(pos) * percentage / 100), int(len(neg) * percentage / 100)


def divide_train_test(pos, neg, cut_pos, cut_neg):
    data_train, train_label = list(pos)[:cut_pos] + list(neg)[:cut_neg], ['pos'] * cut_pos + ['neg'] * cut_neg
    data_test, test_label = list(pos)[cut_pos:] + list(neg)[cut_neg:], ['pos'] * (len(pos) - cut_pos) + ['neg'] * (
        len(neg) - cut_neg)
    return data_train, train_label, data_test, test_label


"L2 normalization_row"


def l2normalize(mat):
    mat = mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor = np.linalg.norm(mat[i].data, 2)
        if not nor == 0:
            for k in mat[i].indices:
                mat[i, k] = mat[i, k] / nor
    return mat


def _topics(res=''):
    # fileB = ['pitsA', 'pitsB', 'pitsC', 'pitsD', 'pitsE', 'pitsF', 'processed_citemap.txt']
    # fileB = ['SE0', 'SE6', 'SE1', 'SE8', 'SE3']
    filepath = '/share/aagrawa8/Data/SE/'
    start_time = time.time()
    #filepath = '/Users/amrit/GITHUB/LDAClassification/dataset/SE/'

    random.seed(1)
    global bounds
    cross_tune = 'no'
    grow_oracle = 'yes'
    data_samples, labellist = readfile1(filepath + str(res) + '.txt')
    split = split_two(corpus=data_samples, label=labellist)
    pos = np.array(split['pos'])
    neg = np.array(split['neg'])

    cut_pos, cut_neg = cut_position(pos, neg, percentage=40)
    ##list of f2 scores
    untuned_lis = []
    tuned_lis = []
    # dictionary containing bestone, time for 1 run, f2
    cross = {}
    # dictionary containing cross, lis,full time
    file = {}
    for folds in range(5):
        start_time1 = time.time()
        pos_shuffle = range(0, len(pos))
        neg_shuffle = range(0, len(neg))
        shuffle(pos_shuffle)
        shuffle(neg_shuffle)
        pos = pos[pos_shuffle]
        neg = neg[neg_shuffle]
        data_train, train_label, data_test, test_label = divide_train_test(pos, neg, cut_pos, cut_neg)
        # stability score format dict, file,lab=score
        # parameter variations (k,a,b), format, list of lists, file,lab=[[k,a,b], Rn score, fscore]
        # final_para_dic={}
        # final paras and scores, file, lab=[[k,a,b],[r, f1]]
        de = DE(F=0.7, CR=0.3, x='rand')

        global max_fitness
        max_fitness = 0
        pop = [[random.randint(bounds[0][0], bounds[0][1]), random.uniform(bounds[1][0], bounds[1][1]),
                random.uniform(bounds[2][0], bounds[2][1])]
               for _ in range(10)]
        v, score, final_para_dic = de.solve(main, pop, iterations=3, bounds=bounds, file=res, term=7,
                                            data_samples=data_train, target=train_label, tune='yes')
        ##score is a list of [jaccard and fscore]
        bestone = [v, score]
        # runtime,format dict, file,=runtime in secs
        l = bestone

        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(data_train + data_test)
        lda1 = lda.LDA(n_topics=int(l[0][0]), alpha=l[0][1], eta=l[0][2], n_iter=200)
        lda1.fit_transform(tf)

        # l2 normalization
        tops = lda1.doc_topic_
        #tops = csr_matrix(tops)
        #tops = l2normalize(tops).toarray()

        split = split_two(corpus=tops, label=np.array(train_label + test_label))
        pos1 = np.array(split['pos'])
        neg1 = np.array(split['neg'])
        data_train, train_label, data_test, test_label = divide_train_test(pos1, neg1, cut_pos, cut_neg)

        split1 = split_two(corpus=data_test, label=np.array(test_label))
        pos1 = np.array(split1['pos'])
        neg1 = np.array(split1['neg'])
        cut_pos1, cut_neg1 = cut_position(pos1, neg1, percentage=50)
        data_grow, grow_label, data_test, test_label = divide_train_test(pos1, neg1, cut_pos1, cut_neg1)

        ## Run with default features
        perc = len(train_label) * 100 / len(train_label + test_label)

        weight_length = int(l[0][0])
        new_bounds = bound * weight_length
        pop1 = [1.0 for _ in range(weight_length)]
        f21 = svmtopics.main(*pop1, data=data_train + data_test, target=train_label + test_label, tune='no',
                             percentage=perc)
        untuned_lis.append(f21)
        time2 = time.time() - start_time1
        bestone.append(time2)
        start_time2 = time.time()

        ## Another DE to find the magic weights
        max_fitness = 0
        pop = [[random.uniform(bound[0][0], bound[0][1]) for _ in range(weight_length)]
               for _ in range(10)]
        perc1 = (len(train_label) + len(grow_label) / 2) * 100 / len(train_label + grow_label)
        v, score, final_para_dic = de.solve(svmtopics.main, pop, iterations=3, bounds=new_bounds,
                                            data=data_train + data_grow, target=train_label + grow_label,
                                            tune='no', percentage=perc1)
        bestone1 = [v, score]

        #testing the modified features.
        f22 = svmtopics.main(*v, data=data_train+data_test, target=train_label + test_label, tune='no',percentage=perc)
        time3 = time.time() - start_time2
        bestone1.append(time3)
        tuned_lis.append(f22)
        cross[folds] = [bestone, bestone1, f21, f22]

        print("\nRuntime for 1 loop of DE termination: --- %s seconds ---\n" % (time2 + time3))
    time1 = time.time() - start_time
    file[res] = [cross, untuned_lis, tuned_lis, time1]
    print(file[res])
    print("\nTotal Runtime: --- %s seconds ---\n" % (time.time() - start_time))
    with open('dump/DE_magic_weights_' + res + '.pickle', 'wb') as handle:
        pickle.dump(file, handle)


bound = [(0.1, 2)]
bounds = [(70, 150), (0.1, 1), (0.01, 1)]
max_fitness = 0
if __name__ == '__main__':
    eval(cmd())
