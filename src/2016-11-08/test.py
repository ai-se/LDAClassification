from __future__ import print_function, division

__author__ = 'amrit'

import sys
import numpy as np
from scipy.sparse import csr_matrix
sys.dont_write_bytecode = True

"L2 normalization_row"
def l2normalize(mat):
    mat=mat.asfptype()
    for i in xrange(mat.shape[0]):
        nor=np.linalg.norm(mat[i].data,2)
        if not nor==0:
            for k in mat[i].indices:
                mat[i,k]=mat[i,k]/nor
    return mat

x1 = np.arange(9.0).reshape((3, 3))
x1=csr_matrix(x1)
print(x1)
matt=l2normalize(x1)
print(matt.toarray())
