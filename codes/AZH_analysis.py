# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:34:25 2020

@author: betty
"""

import math, io, sys,os
import numpy as np
from scipy.sparse import rand
import matplotlib.pyplot as plt
from sklearn import linear_model,tree,svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#
#
#os.chdir(r'D:\research in lab\neighboorhood relationship\Programming\Matlab\feature_extraction')
#
#sys.path.insert(0, os.getcwd()+'\Libfcts')
##from skfeature.function.structure import tree_fs
from tree_fs_customize import tree_fs_customize  as tree_fs

#groupIDs = 

#%%