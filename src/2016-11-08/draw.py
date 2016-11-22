__author__ = 'amrit'

import matplotlib.pyplot as plt
import os, pickle
import operator
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors

if __name__ == '__main__':

    fileB = ['SE0' ,'SE1','SE3','SE6','SE8','cs','diy','photo','rpg','scifi']
    F_final1={}
    current_dic1={}
    para_dict1={}
    time1={}
    path = '/Users/amrit/GITHUB/LDAClassification/src/2016-11-08/dump/'
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            a = os.path.join(root, name)
            with open(a, 'rb') as handle:
                F_final = pickle.load(handle)
                F_final1 = dict(F_final1.items() + F_final.items())
    print(F_final1)

    font = {
        'size': 60}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 10, 'legend.fontsize': 35, 'axes.labelsize': 60, 'legend.frameon': False,
             'figure.autolayout': True}
    plt.rcParams.update(paras)
    X = range(len(fileB))
    plt.figure(num=0, figsize=(25, 15))

    untuned={}
    tuned={}
    Y_tuned = []
    Y_tuned_iqr = []
    Y_untuned=[]
    Y_untuned_iqr=[]

    for file1 in fileB:
        l=[]
        for i in F_final1[file1][2]:
            l.append(i[0])
        Y_tuned.append(np.median(l))
        Y_tuned_iqr.append(np.percentile(l,75)-np.percentile(l,25))
        l1 = []
        for j in F_final1[file1][1]:
            l1.append(j[0])
        Y_untuned.append(np.median(l1))
        Y_untuned_iqr.append(np.percentile(l1, 75) - np.percentile(l1, 25))
    line,=plt.plot(X, Y_untuned,marker='*', markersize=20, label='untuned median')
    plt.plot(X, Y_untuned_iqr,linestyle="-.", markersize=20,color=line.get_color(),label='untuned iqr')
    line,=plt.plot(X, Y_tuned,marker='*', markersize=20, label='tuned median')
    plt.plot(X, Y_tuned_iqr,linestyle="-.", markersize=20,color=line.get_color(),label='tuned iqr')
        #plt.ytext(0.04, 0.5, va='center', rotation='vertical', fontsize=11)
        #plt.text(0.04, 0.5,"Rn (Raw Score)", labelpad=100)
    plt.ylim(0.0, 1.0)
    plt.xticks(X, fileB)
    plt.ylabel("F2 Score", labelpad=30)
    plt.xlabel("Datasets", labelpad=30)
    plt.legend(bbox_to_anchor=(1.07, 1.13), loc=1, ncol=3, borderaxespad=0.1)
    plt.savefig("magic_weights.png")
