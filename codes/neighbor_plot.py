# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:14:31 2017

@author: xli63
"""

import math, io, sys,os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
from scipy.sparse import rand
import scipy.spatial as spatial
from sklearn import linear_model,tree,svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import pylab
os.chdir(r'D:\research in lab\neighboorhood relationship\Programming\Matlab\feature_extraction')


sys.path.insert(0, os.getcwd()+'\Libfcts')
#from skfeature.function.structure import tree_fs
from tree_fs_customize import tree_fs_customize  as tree_fs

#%%
def calculate_NeighborhoodAttraction (dist_ls ,Nei_rad ):
    AttrWeight = 0;
    if  len(dist_ls) > 0:
        for dist in dist_ls:
            AttrWeight = AttrWeight + np.exp (- dist**2/ (2*Nei_rad**2 ))
    return AttrWeight       

def fixRadius_analysis (Coord_Center,Coord_Neighb,Nei_rad = 200):

    Num_Neighbor    = np.zeros(Coord_Center.shape[1])
    AvgDist         = np.zeros_like(Num_Neighbor) + np.nan
    AttrWeight      = np.zeros_like(Num_Neighbor)
    
    center_points   = np.transpose( Coord_Center ) 
    neighbor_points = np.transpose( Coord_Neighb )      
    
    if Coord_Neighb.shape[1]> 0 and Coord_Center.shape[1] > 0 :  # number of neighbors
        # Construc the 
        neighbor_tree   = spatial.cKDTree(neighbor_points)    
        select_neightbor_id_ls = neighbor_tree.query_ball_point(center_points, Nei_rad)
         
        for i , select_neightbor_ids in enumerate( select_neightbor_id_ls):
            Num_Neighbor[i] = len(select_neightbor_ids)
            dist_ls = spatial.distance.cdist( [ center_points[i,:]],
                                               neighbor_points[select_neightbor_ids,:]) [0]  
            AvgDist[i] = dist_ls .mean()
            AttrWeight[i] =  calculate_NeighborhoodAttraction (dist_ls ,Nei_rad )

    return Num_Neighbor, AvgDist, AttrWeight
 
#%%
Tissue_ls = ['LiVPa','vehicle','sham']

#CenterType_ls = ["Neuron"]
#NeighborType_ls = ["Astrocyte","Oligodendrocyte","Microglia"]
save_dir= r"D:\research in lab\neighboorhood relationship\results"

floc = r'D:\research in lab\neighboorhood relationship\datasets\Cropped_Classficiation updates'
CenterType_ls = ["GABAergic_Neuron","NonGABAergic_Neuron","Proliferating_Neuron", "Apoptotic_Neuron","Necrotic_Neuron"]
NeighborType_ls = ['Resting_Astrocyte','Reactive_Astrocyte', 'Proliferating_Astrocyte', 'Apoptotic_Astrocyte',  'Necrotic_Astrocyte', 
               'Myelinating_Oligodendrocyte',  'NonMyelinating_Oligodendrocyte', 'Proliferating_Oligodendrocyte',
               'Apoptotic_Oligodendrocyte', 'Necrotic_Oligodendrocyte', 
               'Resting_Microglia', 'Reactive_Microglia', 'Proliferating_Microglia',
               'Apoptotic_Microglia', 'Necrotic_Microglia']


node_list = CenterType_ls + NeighborType_ls  

#abbr_CenterType_ls = ["GABANeu", "nonGABANeu", "ProfNeu", "ApopNeu","NecroNeu"]
#abbr_NeighborType_ls = ['RestAstro','ReactAstro', 'ProlifAstro', 'ApopAstro',  'NecrAstro', 
#               'MyeliOlig',  'nonMyeliOlig', 'ProlifOlig', 'ApopOlig', 'NecrOlig', 
#               'RestMicrog', 'ReactMicrog', 'ProlifMicrog','ApopMicrog', 'NecrMicrog']

#%%
# get the pair wise name
xv, yv = np.meshgrid( np.arange(len(NeighborType_ls)) , np.arange(len(CenterType_ls)),
                     sparse=False, indexing='ij')
pair_name_ls = []
for n_i,c_i in zip(xv.reshape(-1),yv.reshape(-1),) :
    pair_name_ls.append(NeighborType_ls[n_i] + " in " + CenterType_ls[c_i] )


#%%
tissue_tables = {}
for Tissue in Tissue_ls:
    df_centers = {}
    INDATA = pd.read_excel(os.path.join(floc,Tissue+"_CellTypeStateTable.xlsx"))
    INDATA = INDATA.set_index("ID")      
    
    for CenterType in CenterType_ls:
        center_ID = INDATA.index[INDATA[CenterType]==1]
        Coord_Center = np.array([ INDATA["centroid_x"][center_ID].tolist(),
                                  INDATA["centroid_y"][center_ID].tolist()])   
        Mean_Num_neighbor_ls = []
        Mean_AttrWeight_ls = []
        Mean_AvgDist_ls = []
        Median_Num_neighbor_ls = []
        Median_AttrWeight_ls = []
        Median_AvgDist_ls = []
        
        Sum_Num_neighbor_ls = []
        Sum_AttrWeight_ls = []
        Neighb_ls = []
        Center_ls = []

        for NeighborType in NeighborType_ls:
            neighbor_ID = INDATA.index[INDATA[NeighborType]==1]
            Coord_Neighb =  np.array([ INDATA["centroid_x"][neighbor_ID].tolist(),
                                       INDATA["centroid_y"][neighbor_ID].tolist()])       
            if Coord_Center.shape[1] > 0: # number of center cells is non zero 
                Num_Neighbor, AvgDist, AttrWeight = fixRadius_analysis (Coord_Center,Coord_Neighb,Nei_rad = 200)
            else:
                Num_Neighbor, AvgDist, AttrWeight = [ np.zeros(1),np.zeros(1),np.zeros(1)]
                
            Center_ls.append( CenterType)
            Neighb_ls.append( NeighborType )
            Mean_Num_neighbor_ls.append( Num_Neighbor[np.isnan(Num_Neighbor)==0].mean() )
            Mean_AttrWeight_ls.append( AttrWeight[np.isnan(AttrWeight)==0].mean() )
            Mean_AvgDist_ls.append (AvgDist[np.isnan(AvgDist)==0].mean() )


            Median_Num_neighbor_ls.append( np.median( Num_Neighbor[np.isnan(Num_Neighbor)==0]) )
            Median_AttrWeight_ls.append( np.median(AttrWeight[np.isnan(AttrWeight)==0]) )
            Median_AvgDist_ls.append (np.median(AvgDist[np.isnan(AvgDist)==0]) )

            Sum_Num_neighbor_ls.append( Num_Neighbor[np.isnan(Num_Neighbor)==0].sum() )
            Sum_AttrWeight_ls.append( AttrWeight[np.isnan(AttrWeight)==0].sum() )

        df = pd.DataFrame({'neighb':Neighb_ls, 'center':Center_ls, 
                           'Mean_AvgDist':Mean_AvgDist_ls,'Mean_AttrWeight': Mean_AttrWeight_ls,'Mean_Num_neighbor': Mean_Num_neighbor_ls,
                           'Median_AvgDist':Median_AvgDist_ls,'Median_AttrWeight': Median_AttrWeight_ls,'Median_Num_neighbor': Median_Num_neighbor_ls,
                           'Sum_Num_neighbor': Sum_Num_neighbor_ls,'Sum_AttrWeight':Sum_AttrWeight_ls
                           })
        df_centers[CenterType] = df
    tissue_tables[Tissue]= df_centers

#%%
# https://stackoverflow.com/questions/44291155/plotting-two-distance-matrices-together-on-same-plot?noredirect=1&lq=1
def triatpos(pos=(0,0), rot=0,vmin=0,vmax=1):
    r = np.array([[-1,-1],[1,-1],[1,1],[-1,-1]])*.5
    rm = [[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot))],
           [np.sin(np.deg2rad(rot)),np.cos(np.deg2rad(rot)) ] ]
    r = np.dot(rm, r.T).T
    r[:,0] += pos[0]
    r[:,1] += pos[1]
    return r

def triamatrix(a,  rot=0, cmap=plt.cm.viridis, **kwargs):
    segs = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            segs.append(triatpos((j,i), rot=rot) )
    col = mpl.collections.PolyCollection(segs, cmap=cmap, **kwargs)
    col.set_array(a.flatten())
#        ax.add_collection(col)
    return col
#%%    

compareTissues =  [[1,3],[1,2], [2,3]]
Features = df_centers[CenterType].columns.tolist()[:-2] 
Features = ["Mean_Num_neighbor","Median_Num_neighbor" ]

#%%
''' plt the comparision of absolute value '''
for feature in Features:                            
    fig, axes= plt.subplots(1,3 ,figsize = (16,8))    
    fig.subplots_adjust(bottom=0.16)
    plt.suptitle("Comparision Map of " + feature)
    for ct_i , compareTissue in enumerate(  compareTissues )  :   
        # get the number of neighbor sheet
        
        t_name0 = Tissue_ls[ compareTissue[0] - 1 ]
        numNeighbor_array0 = np.zeros ((15,5))
        df_centers = tissue_tables [t_name0]
        for c_i, CenterType in enumerate(CenterType_ls):
            numNeighbor_array0[:,c_i] = df_centers[CenterType][feature]   
#        triamatrix(numNeighbor_array0, 0)    
        
        t_name1 = Tissue_ls[ compareTissue[1] - 1 ]
        numNeighbor_array1 = np.zeros ((15,5))
        df_centers = tissue_tables [t_name1 ]
        for c_i, CenterType in enumerate(CenterType_ls):
            numNeighbor_array1[:,c_i] = df_centers[CenterType][feature]   
                    
        vmin = np.nanmin([numNeighbor_array0,numNeighbor_array1])
        vmax = np.nanmax([numNeighbor_array0,numNeighbor_array1])
        
        ax = axes[ct_i]    
        
        im1 = ax.imshow(numNeighbor_array0, vmin=vmin,vmax=vmax,cmap="binary")          # top left
        im2 = triamatrix(numNeighbor_array1, rot=90,cmap="binary")   # top bottem
        im2.set_clim(vmin,vmax)
        ax.add_collection(im2)            
        plt.sca(ax) 
        
        plt.xticks([0, 1, 2, 3, 4],  CenterType_ls, rotation=45 )
        if ct_i==0:
            plt.yticks(range(15), NeighborType_ls)
        else:
            plt.yticks(range(15), [])

        plt.title (t_name0 + " VS " + t_name1  )
        plt.colorbar(im1)    

    plt.savefig(os.path.join(save_dir,"CompareMap of " + feature + "png" ))               
               
                                            
#%%   
''' plt the different values '''
for feature in  Features:                            
    fig, axes= plt.subplots(1,3 ,figsize = (16,8))
    fig.suptitle("Difference map of " + feature)
    for ct_i , compareTissue in enumerate(  compareTissues )  :   
        # get the number of neighbor sheet
        
        t_name0 = Tissue_ls[ compareTissue[0] - 1 ]
        numNeighbor_array0 = np.zeros ((15,5))
        df_centers = tissue_tables [t_name0]
        for c_i, CenterType in enumerate(CenterType_ls):
            numNeighbor_array0[:,c_i] = df_centers[CenterType][feature]   
        triamatrix(numNeighbor_array0, 0)    
        
        t_name1 = Tissue_ls[ compareTissue[1] - 1 ]
        numNeighbor_array1 = np.zeros ((15,5))
        df_centers = tissue_tables [t_name1 ]
        for c_i, CenterType in enumerate(CenterType_ls):
            numNeighbor_array1[:,c_i] = df_centers[CenterType][feature]   
        
            
        diffmap = numNeighbor_array0 - numNeighbor_array1
        max_abs = max( abs(diffmap.max()),abs(diffmap.min()))
        
        ax = axes[ct_i]            
        im1 = ax.imshow(numNeighbor_array0- numNeighbor_array1,cmap='bwr', vmin=-max_abs, vmax=max_abs)
        
        plt.sca(ax)    
        plt.xticks([0, 1, 2, 3, 4],  CenterType_ls, rotation='45')
        if ct_i==0:
            plt.yticks(range(15), NeighborType_ls)
        else:
            plt.yticks(range(15), [])
        plt.title (t_name0 + " - " + t_name1  )
        plt.colorbar(im1)                  
        
    plt.subplots_adjust(left=0.1, right=0.9, top=0.91, bottom=0.2)
    plt.savefig(os.path.join(save_dir,"DiffMap of " + feature + "png" ))               


#%%
''' plt the different values '''

for feature in  Features:                            
    fig, axes= plt.subplots(1,3,figsize=(16,8))    
    fig.suptitle("Difference Bar of " + feature)
    for ct_i , compareTissue in enumerate(  compareTissues )  :   
        # get the number of neighbor sheet
        
        t_name0 = Tissue_ls[ compareTissue[0] - 1 ]
        numNeighbor_array0 = np.zeros ((15,5))
        df_centers = tissue_tables [t_name0]
        for c_i, CenterType in enumerate(CenterType_ls):
            numNeighbor_array0[:,c_i] = df_centers[CenterType][feature]   
        triamatrix(numNeighbor_array0, 0)    
        
        t_name1 = Tissue_ls[ compareTissue[1] - 1 ]
        numNeighbor_array1 = np.zeros ((15,5))
        df_centers = tissue_tables [t_name1 ]
        for c_i, CenterType in enumerate(CenterType_ls):
            numNeighbor_array1[:,c_i] = df_centers[CenterType][feature]           
            
        diffmap = numNeighbor_array0- numNeighbor_array1
        max_abs = max( abs(diffmap.max()),abs(diffmap.min()))
        
        # SORT CALUES                
        reshaped_value =  diffmap.reshape(-1)
        sored_reshaped_value = np.sort(reshaped_value)


        assorted_ids = np.argsort(reshaped_value)
        assorted_ids = assorted_ids[sored_reshaped_value!=0]
        assorted_names =  [pair_name_ls[i] for i in assorted_ids]
        
        len_neg = (sored_reshaped_value<0).sum()
        
        ax = axes[ct_i]  
        sored_reshaped_value = sored_reshaped_value[sored_reshaped_value!=0]
        ax.barh(np.arange(len(assorted_ids)),sored_reshaped_value,color ="r")
        ax.barh(np.arange(len(assorted_ids))[:len_neg],sored_reshaped_value[:len_neg],color ="b")
        plt.sca(ax)    

        plt.yticks(np.arange(len(assorted_ids)), assorted_names)

        plt.title (t_name0 + " - " + t_name1  )
        plt.grid(zorder=0)
        
    plt.tight_layout()
    plt.subplots_adjust( top=0.91)
        
    plt.savefig(os.path.join(save_dir,"NonZeroDiffBar of " + feature + ".png" ))               


#%%
def my_circle_scatter_radii(axes, x_array, y_array, r, **kwargs):   # drar neighborhood circles
    
    for (x, y) in zip(x_array, y_array):
        circle = pylab.Circle((x,y), radius=r, **kwargs)
        axes.add_patch(circle)
    return axes

#%%    visulize the chaning
def plot_change(compair_tissues, c_type,n_type):
    fig, axes= plt.subplots(1,2,figsize=(10,4),dpi=200)    
    for ti, Tissue in enumerate( compair_tissues):        
        INDATA = pd.read_excel(os.path.join(floc,Tissue+"_CellTypeStateTable.xlsx"))    
        center_ID = INDATA.index[INDATA[c_type]==1]
        neighbor_ID = INDATA.index[INDATA[n_type]==1]
        
        ax = axes[ti]
        my_circle_scatter_radii( ax, INDATA["centroid_x"][center_ID], INDATA["centroid_y"][center_ID], 
                                r= 200,color="b", alpha=0.05)

        cellplt = ax.scatter ( INDATA["centroid_x"],INDATA["centroid_y"],1,color="gray",alpha=0.1)
        cplt = ax.scatter ( INDATA["centroid_x"][center_ID],INDATA["centroid_y"][center_ID], 1,color="b")
        nplt = ax.scatter ( INDATA["centroid_x"][neighbor_ID],INDATA["centroid_y"][neighbor_ID], 0.5,color="r")

        plt.sca(ax) 
        plt.legend([cellplt,cplt,nplt], ["All Cells",c_type,n_type])        
        plt.title(Tissue)
        plt.tight_layout()        
        plt.xticks([])
        plt.yticks([])
    fig.savefig(os.path.join(save_dir,"plot Vis " + c_type + ' in' + n_type + ".png" ))         
#%%
compair_tissues = [ "LiVPa", "vehicle"]    
c_type , n_type = ["NonGABAergic_Neuron","Resting_Microglia"]
plot_change(compair_tissues, c_type,n_type)


compair_tissues = [ "vehicle", "sham"]    
c_type , n_type = ["NonGABAergic_Neuron","Resting_Microglia"]
plot_change(compair_tissues, c_type,n_type)

    
compair_tissues = [ "LiVPa", "sham"]        
c_type , n_type = ["Necrotic_Neuron","Apoptotic_Oligodendrocyte"]
plot_change(compair_tissues, c_type,n_type)        


    
#%%    

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


def runFeatureSelectionCompare (data, label):
    ## Construct GroupID_tree
    data_id = []
    data_id.append ( [ -1,-1,1 ])
    for status_id  in range(0,15):
    #    print (status_id)
#            bottom_idx = [1+ status_id *5 , 5 + status_id*5, 1]
        bottom_idx = [status_id *5 , 4 + status_id*5, 1]
        
        data_id.append (bottom_idx)
        
    for status_id  in range(0,5):
    #    print (status_id)
#            middle_idx = [1+ status_id *15 , 15 + status_id*15, 1]
        middle_idx = [ status_id *15 , 14 + status_id*15, 1]

        data_id.append (middle_idx)
    
    data_id_array = np.array(data_id).T.astype(int)
#    print (data_id)

    # lasso
    # perform feature selection and obtain the feature weight of all the features

    w_treeLasso, obj, value_gamma = tree_fs(X = data , y = label, z = 0.01, idx = data_id_array, max_iter = 5000, verbose=False)
    selected_featureBoolean =   (abs(w_treeLasso)>1) * 1
                    
    data_id_flat= np.array([[-1,-1,1],[0,74,1]]).T
    w_Lasso, obj, value_gamma = tree_fs(X = data , y = label, z = 0.01, idx = data_id_flat, max_iter = 5000, verbose=False)
    selected_featureBoolean =   (abs(w_treeLasso)>1) * 1
    
    TreeLasso_reshaped = np.reshape(  abs(w_treeLasso) ,[15,5], order = 'F') /10

    return w_Lasso, selected_featureBoolean

#%%    LASSO comparison
    ''' Occasions  '''
    #Compare it for ( LiVPa 1 VS sham3)( LiVPa 1 VS vehicle2)(Vehicle2 VS Sham 3 )
    aa= np.arange(75)
    np.reshape(  abs(aa) ,[15,5], order = 'F')

    
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
    runFeatureSelectionCompare (data = data , 
                                    label = label)        


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
       
#        plt.scatter(np.arange(len(selected_pairs)),np.mean(N_Data,0), marker='o', color='g')        
#        plt.scatter(np.arange(len(selected_pairs)),np.mean(N_Data,0) +np.std(N_Data,0) , marker='_', color='g')        
#        plt.scatter(np.arange(len(selected_pairs)),np.mean(N_Data,0) -np.std(N_Data,0) , marker='_', color='g')    
#        plt.errorbar(np.arange(len(selected_pairs)), np.mean(N_Data,0),np.std(N_Data,0),fmt=None,marker='.',c='g')    
            
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
            
             


if __name__ == '__main__':
    main()