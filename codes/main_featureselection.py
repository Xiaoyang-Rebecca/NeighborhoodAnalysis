# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:14:31 2017

@author: xli63
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
#
#
#tree_fs(X = data, y = label, z = 0.2, idx = groupIDs
#        , 'True')



def visialize(obj,w = []):
    
    
    plt.figure()
    x = np.arange(0,np.shape(obj)[0])
    
    plt.subplot(2, 1, 1)
#    plt.plot(x,obj,'.-')    
    plt.plot(x,obj)    

    plt.xlabel('iters')
    plt.ylabel('obj value')
    plt.title('tree structured group lasso regularization')
      
    
    if w!=[]:        
        plt.subplot(2, 1, 2)
        plt.imshow(np.array([abs(w),abs(w)]))
        
def classification(classifier_Name,data,label,predictData = []):
    if classifier_Name=='LogisticRegression':
        clf = LogisticRegression(C=1,penalty='l1', tol=0.0001)
    elif classifier_Name=='LinearSVC':
        clf = svm.LinearSVC()
    elif classifier_Name=='KSVM':
        clf = svm.SVC( kernel='rbf')
    elif classifier_Name=='MLP':
        clf = MLPClassifier()
    else: #classifier_Name=='DecisionTree':
        clf = tree.DecisionTreeClassifier()
                 
    clf.fit(X = data, y = label )
    
    if predictData == []:
        label_predict = clf.predict(X = data)
    else:
        label_predict = clf.predict(X = predictData)


    accuracy = np.sum( label_predict ==label )/len(label)            
            
    return accuracy,label_predict

#%%

def runFeatureSelectionCompare (data, label,CenterName,NeighborName):
    ## Construct GroupID_tree
            
    data_id = []
    data_id.append ( [ -1,-1,1 ])
    for status_id  in range(0,len(NeighborName)):
    #    print (status_id)
#            bottom_idx = [1+ status_id *5 , 5 + status_id*5, 1]
        bottom_idx = [status_id *len(CenterName) , len(CenterName)-1 + status_id*len(CenterName), 1]
        data_id.append (bottom_idx)
        
    for status_id  in range(0,len(CenterName)):
    #    print (status_id)
#            middle_idx = [1+ status_id *15 , 15 + status_id*15, 1]
        middle_idx = [ status_id *len(NeighborName) , len(NeighborName)-1 + status_id*len(NeighborName), 1]
        data_id.append (middle_idx)
    
    data_id_array = np.array(data_id).T.astype(int)
    print (data_id)

    # lasso
    # perform feature selection and obtain the feature weight of all the features

    w_treeLasso, obj, value_gamma = tree_fs(X = data , y = label, z = 0.01, 
                                            idx = data_id_array, max_iter = 5000, verbose=False)
    selected_featureBoolean =   (abs(w_treeLasso)>1) * 1
            
    
    data_id_flat= np.array([[-1,-1,1],[0,len(CenterName)*len(NeighborName)-1,1]]).T
    w_Lasso, obj, value_gamma = tree_fs(X = data , y = label, z = 0.01, 
                                        idx = data_id_flat, max_iter = 5000, verbose=False)
    selected_featureBoolean =   (abs(w_treeLasso)>1) * 1

    return w_Lasso, w_treeLasso
#%%
def vis_lasso(data,label,w_Lasso,w_treeLasso, CenterName,NeighborName):

    P_Data = data[label==1,:]
    N_Data = data[label==-1,:]
            
    plt.figure()
    I = np.argsort(abs(w_treeLasso))[::-1]  # decent order        
    TreeLasso_reshaped = np.reshape(  abs(w_treeLasso) ,[len(NeighborName),len(CenterName)], order = 'F') /10
    plt.imshow(TreeLasso_reshaped)        
    plt.xticks( range(len(CenterName)),  CenterName, rotation='vertical')
    plt.yticks(range(len(NeighborName)), NeighborName)
    plt.title ("Lasso Factors")
    plt.xticks(rotation=45)
    plt.colorbar()
        
    plt.figure()
    plt.subplot (2,1,1)
    plt.bar(np.arange(len(w_treeLasso)),abs(w_treeLasso))
    
    plt.subplot (2,1,2)
    plt.scatter(np.arange(len(w_treeLasso)),np.mean(P_Data,0), marker='o', color='r')           
    plt.scatter(np.arange(len(w_treeLasso)),np.mean(P_Data,0) +np.std(P_Data,0) , marker='_', color='r')        
    plt.scatter(np.arange(len(w_treeLasso)),np.mean(P_Data,0) -np.std(P_Data,0) , marker='_', color='r')        
    plt.errorbar(np.arange(len(w_treeLasso)), np.mean(P_Data,0),np.std(P_Data,0),fmt=None,marker='.',c='r')
    
    plt.scatter(np.arange(len(w_treeLasso)),np.mean(N_Data,0), marker='o', color='g')        
    plt.scatter(np.arange(len(w_treeLasso)),np.mean(N_Data,0) +np.std(N_Data,0) , marker='_', color='g')        
    plt.scatter(np.arange(len(w_treeLasso)),np.mean(N_Data,0) -np.std(N_Data,0) , marker='_', color='g')    
    plt.errorbar(np.arange(len(w_treeLasso)), np.mean(N_Data,0),np.std(N_Data,0),fmt=None,marker='.',c='g')        

    
    selected_pairs = I [:len(CenterName)] 
    
    plt.figure()
    plt.subplot (2,1,1)
    plt.bar(np.arange(len(selected_pairs)),abs(w_treeLasso)[selected_pairs])
           
    
    P_Data = P_Data[:,selected_pairs]
    N_Data = N_Data[:,selected_pairs]

    plt.subplot (2,1,2)
    plt.scatter(np.arange(len(selected_pairs)),np.mean(P_Data,0), marker='o', color='r')           
    plt.scatter(np.arange(len(selected_pairs)),np.mean(P_Data,0) +np.std(P_Data,0) , marker='_', color='r')        
    plt.scatter(np.arange(len(selected_pairs)),np.mean(P_Data,0) -np.std(P_Data,0) , marker='_', color='r')        
    plt.errorbar(np.arange(len(selected_pairs)), np.mean(P_Data,0),np.std(P_Data,0),fmt=None,marker='.',c='r')
    
    plt.scatter(np.arange(len(selected_pairs)),np.mean(N_Data,0), marker='o', color='g')        
    plt.scatter(np.arange(len(selected_pairs)),np.mean(N_Data,0) +np.std(N_Data,0) , marker='_', color='g')        
    plt.scatter(np.arange(len(selected_pairs)),np.mean(N_Data,0) -np.std(N_Data,0) , marker='_', color='g')    
    plt.errorbar(np.arange(len(selected_pairs)), np.mean(N_Data,0),np.std(N_Data,0),fmt=None,marker='.',c='g')    
        
#%%

def main():
#    n_samples = 50    # specify the number of samples in the simulated data
#    n_features = 100    # specify the number of features in the simulated data
#
#    # simulate the dataset
#    X = np.random.rand(n_samples, n_features)
#
#    # simulate the feature weight
#    w_orin = rand(n_features, 1, 1).toarray()
#    w_orin[0:50] = 0
#
#    # obtain the ground truth of the simulated dataset
#    noise = np.random.rand(n_samples, 1)
#    y = np.dot(X, w_orin) + 0.01 * noise
#    y = y[:, 0]
#
#
#    z = 0.01  # specify the regularization parameter of regularization parameter of L2 norm for the non-overlapping group
#
#    # specify the tree structure among features
#    idx = np.array([[-1, -1, 1], [1, 20, np.sqrt(20)], [21, 40, np.sqrt(20)], [41, 50, np.sqrt(10)],
#                    [51, 70, np.sqrt(20)], [71, 100, np.sqrt(30)], [1, 50, np.sqrt(50)], [51, 100, np.sqrt(50)]]).T
#    idx = idx.astype(int)
        
    fileName = 'InputPacking2_Data.txt'
    data = np.loadtxt(fileName, delimiter=',')  
#    ## data normalization
#    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
#    scaler.fit(data)
#    data = scaler.transform(data)
        
    fileName = 'InputPacking2_Label.txt'
    label = np.loadtxt(fileName, delimiter=',')  
    


#        #''' Classification Result Summary'''
##        for classifier_Name in ['LogisticRegression','LinearSVC','KSVM','MLPClassifier','DecisionTree']:                               
#        for classifier_Name in ['LogisticRegression']:                               
#                
#            ##    Before feature selection
#            accuracy_before, __ = classification(classifier_Name,
#                                             data = data,
#                                             label = label)           
#            ##    After feature selection
#            
#            selected_featureID = selected_featureBoolean* ( np.arange(75) +1 ) 
#            selected_featureID = selected_featureID[selected_featureID>0]
#            selectedData = data[:,selected_featureID-1]
#            accuracy_after, __ = classification(classifier_Name ,
#                                             data = selectedData,
#                                             label = label)        
#            print (classifier_Name,accuracy_before,accuracy_after)
            
            
    ''' Occasions  '''
    #Compare it for ( LiVPa 1 VS sham3)( LiVPa 1 VS vehicle2)(Vehicle2 VS Sham 3 )
    for comparingLabel in [[1,3],[1,2],[2,3]]:
#    for comparingLabel in [[1,3]]:
        print (comparingLabel)
        ComparingID_1 = np.logical_or(label==comparingLabel[0], label== comparingLabel[1])
        data_Occ1 = data[ComparingID_1,:]
        label_1 = label[ComparingID_1]
        label_1[label_1==comparingLabel[0]]=1
        label_1[label_1==comparingLabel[1]]=-1
        runFeatureSelectionCompare (data = data_Occ1 , 
                                    label = label_1)
        
        
    #Compare it for ( LiVPa 1 VS Vehicle2 sham3)
    CenterName = ['GABAergic_Neuron','NonGABAergic_Neuron','Proliferatiing_Neuron','Apoptotic_Neuron','Necrotic_Neuron']
    NeighborName = ['Resting_Astrocyte','Reactive_Astrocyte'	,'Proliferatiing_Astrocyte'	,'Apoptotic_Astrocyte'	,
                        'Necrotic_Astrocyte'	,'Myelinating_Oligodendrocyte',	'NonMyelinating_Oligodendrocyte',	
                        'Proliferatiing_Oligodendrocyte'	,'Apoptotic_Oligodendrocyte',	'Necrotic_Oligodendrocyte',	
                        'Resting_Microglia',	'Reactive_Microglia'	,'Proliferatiing_Microglia'	,'Apoptotic_Microglia',	'Necrotic_Microglia']

    runFeatureSelectionCompare (data = data , 
                                    label = label)

if __name__ == '__main__':
    main()