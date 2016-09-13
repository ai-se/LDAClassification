__author__ = 'amrit'

import matplotlib.pyplot as plt
import os, pickle
import operator
import numpy as np
import matplotlib.cm as cmx
import matplotlib.colors as colors

if __name__ == '__main__':

    fileB = ['SE0' ,'SE6']#,'SE8','SE3','SE1']
    '''F_final1={}
    current_dic1={}
    para_dict1={}
    time1={}
    path = '/home/amrit/GITHUB/LDAClassification/src/2016-09-05/dump/words/untuned/'
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            a = os.path.join(root, name)
            with open(a, 'rb') as handle:
                F_final = pickle.load(handle)
                F_final1 = dict(F_final1.items() + F_final.items())
                #current_dic = pickle.load(handle)
                #current_dic1 = dict(current_dic1.items() + current_dic.items())
                #para_dict = pickle.load(handle)
                #para_dict1 = dict(para_dict1.items() + para_dict.items())
                #time = pickle.load(handle)
                #time1 = dict(time1.items() + time.items())
    print(F_final1)'''
    temp={'200': [0.5957446808510638, 0.64015151515151514, 0.64150943396226412, 0.62068965517241381, 0.60500963391136797, 0.62357414448669202, 0.62307692307692308, 0.61832061068702293, 0.62476190476190474, 0.61420345489443384, 0.60500963391136808, 0.60384615384615381, 0.61923076923076914, 0.64150943396226412, 0.63377609108159383, 0.6212121212121211, 0.61685823754789271, 0.64015151515151514, 0.61003861003861004, 0.61538461538461542, 0.61567877629063106, 0.6195028680688337, 0.62595419847328237, 0.62068965517241381, 0.62213740458015276], 'SE0': [0.64343163538873993, 0.65608465608465605, 0.6487935656836461, 0.65608465608465605, 0.6507936507936507, 0.66840731070496084, 0.65775401069518713, 0.64171122994652396, 0.63806970509383376, 0.64893617021276606, 0.66137566137566139, 0.66137566137566139, 0.6507936507936507, 0.63611859838274931, 0.64533333333333331, 0.59052924791086348, 0.59890109890109888, 0.65240641711229941, 0.66315789473684217, 0.65957446808510634, 0.5842696629213483, 0.62330623306233057, 0.6507936507936507, 0.65066666666666673, 0.65782493368700268]}
    topics=[10,20,40,80,200]
    for file1 in fileB:
        print(np.median(temp[file1]), np.percentile(temp[file1],75)-np.percentile(temp[file1],25))
    '''
    font = {
            'size'   : 60}

    plt.rc('font', **font)
    paras={'lines.linewidth': 10,'legend.fontsize': 35, 'axes.labelsize': 60, 'legend.frameon': False,'figure.autolayout': True}
    plt.rcParams.update(paras)
    X = range(len(labels))
    plt.figure(num=0, figsize=(25, 15))

    #l=['F3CR7P30','F7CR3P30','F3CR7P10','F7CR3P10']
    for file1 in fileB:
        #for s in l:
        Y_tuned=[]
        Y_untuned=[]
        for lab in labels:

            #print(tuned_FCR_vem_py['pitsA'][lab][file1])
            Y_tuned.append(tuned_spark_vem[file1][lab])
            Y_untuned.append(np.median(untuned_spark_vem[file1][lab]))
            #print(Y_tuned)
        line, = plt.plot(X, Y_tuned,marker='o', markersize=20, label='Tuned '+file1)
        plt.plot(X, Y_untuned, linestyle="-.", color=line.get_color(), marker='*', markersize=20, label='Untuned '+file1)
        #plt.ytext(0.04, 0.5, va='center', rotation='vertical', fontsize=11)
        #plt.text(0.04, 0.5,"Rn (Raw Score)", labelpad=100)
    plt.ylim(-0.1,1.1, )
    plt.xticks(X, labels)
    plt.ylabel("Rn (Raw Score)", labelpad=30)
    plt.xlabel("n (No. of terms overlap)",labelpad=30)
    plt.legend(bbox_to_anchor=(0.35, 0.8), loc=1, ncol = 1, borderaxespad=0.)
    plt.savefig("spark1" + ".png")'''
