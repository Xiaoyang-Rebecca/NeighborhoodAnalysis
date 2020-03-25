# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:29:26 2019

@author: xli63
"""

import sys, os
import csv
import math
import numpy as np
import itertools 

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import scipy
import random
import multiprocessing
from multiprocessing.pool import ThreadPool
import matplotlib
import pylab
#     # Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def kde_dist(center_coords,neighbor_coords, RMax):
    # densitiy weighted distance of target point (neighbor) to task point (center)
    kde_d = np.exp(- ( ( center_coords[0]- neighbor_coords[0] )** 2 +  \
                       ( center_coords[1]- neighbor_coords[1] )** 2  ) /  ( 2* RMax ** 2) )
    return kde_d

def candidateExpand (candTable_kde,CPSet_ls ,opt = "min"):
    '''update the candidate table, consider all the type combination cases    '''    
    #Construct all subsets of type set 
    for CP in CPSet_ls:
        candiate_types = list(CP)
        column_name =  '-'.join(candiate_types)   #"_"indicator of number of types in the candiatate set
        if opt == "min" :
            candTable_kde[column_name] = candTable_kde[candiate_types].min(axis=1)
        elif opt == "max" :
            candTable_kde[column_name] = candTable_kde[candiate_types].max(axis=1)
    return candTable_kde

def instance_cliques(ele_ls):    # [definition5] in ref[1]
    ''' Generate all the clipques (full set) containing elements in ele_ls'''
    # e.g.
    # input:   ele_ls = [A,B,C]
    # output: CPSet_ls = [(A),(B),(C),(A,B),(A,C),(B,C),(A,B,C)]
    CPSet_ls = []
    for n in range (len(ele_ls)):  
        CPSet = list( itertools.combinations(ele_ls, n+1) )
        for CP in CPSet:
            CPSet_ls.append(CP)            
    return CPSet_ls

def attractionDefine (Ptable,center_type,neighbor_type_ls, RMax,index_opt, cliques=False)  :    
    
    cCountTable = pd.DataFrame(Ptable[neighbor_type_ls].sum(axis=0)).transpose() # count total numer of candiates for each subset 
    if cliques == True:
        CPSet_ls    = instance_cliques(neighbor_type_ls)                             # generate all type of candidate subsets
        cCountTable = candidateExpand (cCountTable,CPSet_ls,"max")                   # max number of instance of each type
        
    cTable = Ptable[Ptable[center_type]== 1]
    attraction_dict ={}    
    
    for ID in cTable.index.tolist():
        center = cTable.loc[ID]
        circular_area =  ( center["centroid_x"] - Ptable["centroid_x"]  ) **2 + \
                       ( center["centroid_y"] - Ptable["centroid_y"]  ) **2    <= RMax **2
        tTable = Ptable[circular_area == True]   # task instance set T / transaction table

        if  index_opt is not  "kde":    ## General Index for neighbor instance ##
            candTable = tTable[neighbor_type_ls]
            if cliques == True:
                candTable = candidateExpand (candTable,CPSet_ls, "min")                    
            attraction_dict [ID]    = np.array( candTable.sum() )
            import pdb;pdb.set_trace()
            
        else:                          ## Weighted Index for neighbor instance ##
            dist_weight = [ kde_dist( [ center["centroid_x"], center["centroid_y"] ] , 
                                     [tTable.loc[t_id,"centroid_x"],tTable.loc[t_id,"centroid_y"] ],  RMax ) \
                           for t_id in tTable.index ]     # [definition 2] in Ref[1] 
            # Generate candidate table
            candTable_data = np.array( tTable[neighbor_type_ls] )  * \
                          np.tile( np.array( dist_weight), [len(neighbor_type_ls),1]).transpose()      
            tTable["dist_weight"] = dist_weight
            # The cumulative weight of the target instances related to the task instances , [definition 3] in Ref[1] 
            candTable_kde = pd.DataFrame(candTable_data, columns = neighbor_type_ls)             
            if cliques == True:
                candTable_kde = candidateExpand (candTable_kde,CPSet_ls,"min")            
            attraction_dict [ID]= candTable_kde.sum(axis=0)      
    print ("ID=",ID)

    attractionTable = pd.DataFrame.from_dict(attraction_dict, orient = 'index')  
    attractionTable.columns=cCountTable.columns
    
    attraction_index = np.array(attractionTable.sum()) / np.array(cCountTable)       # candiate set
    attraction_index = pd.DataFrame(attraction_index, columns = cCountTable.columns) 
    
    return attractionTable,attraction_index


def my_circle_scatter_radii(axes, x_array, y_array, r, **kwargs):   # drar neighborhood circles
    
    for (x, y) in zip(x_array, y_array):
        circle = pylab.Circle((x,y), radius=r, **kwargs)
        axes.add_patch(circle)
    return axes

#%%
def my_arrow_radii(axes, centers, neighbors, r, **kwargs):   # drar neighborhood circles
    x0_array,y0_array,x1_array, y1_array= [],[],[],[]    # storage all the center-neighbor pairs
    
    for c_i in centers.index:
        center = centers.loc[c_i]
        circular_area =  ( center["centroid_x"] - neighbors["centroid_x"]  ) **2 + \
                         ( center["centroid_y"] - neighbors["centroid_y"]  ) **2    <= r **2
        neighbor =  neighbors[circular_area == True]                           # get the neighbor for this one center
        if len(neighbor) > 0:
            for n_i in neighbor.index:
                x0_array.append(center["centroid_x"])
                y0_array.append(center["centroid_y"])
                x1_array.append(neighbor["centroid_x"].loc[n_i])
                y1_array.append(neighbor["centroid_y"].loc[n_i])                
    for (x0, y0,x1,y1) in zip(x0_array, y0_array,x1_array, y1_array):        
        axes.arrow (x0,y0, x1-x0, y1-y0, **kwargs)   #arrow(x, y, dx, dy, **kwargs)[source]
    return axes

#%%
def scatter_neighbor_plot( cellType_df, center_type ,neighbor_type ,tissue_ls= ["leftCrop", "rightCrop"],title_ls=None,
                          c_color = "b",n_color = "r"):        
    #  visulize the chaning
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1,2,figsize=(10,4),dpi=200)    
    if title_ls is None:
        title_ls = tissue_ls
    for t_i,tissue in enumerate(tissue_ls):        
        INDATA = cellType_df.query(tissue)
        center_ID = INDATA.index[INDATA[center_type]==1]
        neighbor_ID = INDATA.index[INDATA[neighbor_type]==1]
        
        ax = axes[t_i]
        ax.set_title( tissue +":"+ title_ls[t_i])
#        my_circle_scatter_radii( ax, INDATA["centroid_x"][center_ID], INDATA["centroid_y"][center_ID], 
#                                r= 200,color="b", alpha=0.05)
        
        my_arrow_radii( ax, centers = INDATA[["centroid_x","centroid_y"]].loc[center_ID], 
                                     neighbors =  INDATA[["centroid_x","centroid_y"]].loc[neighbor_ID], 
                                r= 200, linestyle="--",linewidth = 0.5 )
       
        cellplt = ax.scatter ( INDATA["centroid_x"],INDATA["centroid_y"],0.5,color="gray",alpha=0.1)    # plot all cells
        cplt = ax.scatter ( INDATA["centroid_x"][center_ID],INDATA["centroid_y"][center_ID], 0.5,color=c_color)
        nplt = ax.scatter ( INDATA["centroid_x"][neighbor_ID],INDATA["centroid_y"][neighbor_ID], 0.5,color=n_color)
        ax.axis("equal")
        ax.set_axis_off()
        ax.axis("tight")  # gets rid of white border
        ax.legend([cellplt,cplt,nplt], ["All Cells",center_type,neighbor_type])                
        
    plt.tight_layout()   

    return fig