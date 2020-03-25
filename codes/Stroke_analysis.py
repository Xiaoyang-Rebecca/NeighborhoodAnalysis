# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:56:48 2019

@author: xli63
"""
import os,sys
import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
sys.path.insert(0, os.getcwd()+'\Libfcts')
import utils 

from main_featureselection import runFeatureSelectionCompare,vis_lasso
    #%%
# load phenotype
if os.path.exists( os.path.abspath("../data/Storke_celltype_table.csv")):
    cellType_df = pd.read_csv( os.path.abspath("../data/Storke_celltype_table.csv"))

else:    
    df = pd.read_csv(os.path.abspath("../data/Storke_classification_table.csv"))
    
    cellType_df = pd.DataFrame ({"ID":df["ID"],
                             "centroid_x":df["centroid_x"],
                             "centroid_y":df["centroid_y"],
                             "Neuron":df["NeuN"],
                             "GABAergic_Neuron":df["NeuN"]*df["Parvalbumin"],
                             "NonGABAergic_Neuron":df["NeuN"]*( 1-df["Parvalbumin"]),                                 
                             "Astrocyte":df["S100B"],
                             "Reactive_Astrocyte": df["S100B"] * df["GFAP"],
                             "Resting_Astrocyte": df["S100B"] * (1-df["GFAP"]),                         
                             "Microglia":df["IBA1"],
                             "Reactive_Microglia": df["IBA1"] * df["TomatoLectin"],
                             "Resting_Microglia": df["IBA1"] * (1-df["TomatoLectin"]) } )
    cellType_df.to_csv(os.path.abspath("../data/Storke_celltype_table.csv"))
    
cellType_df= cellType_df.set_index("ID")


cellType_cl = {
             "Neuron":"green",
             "GABAergic_Neuron":"green",
             "NonGABAergic_Neuron":"blue",                                 
             "Astrocyte":"yellow",
             "Reactive_Astrocyte": "orange",
             "Resting_Astrocyte": "yellow",                         
             "Microglia":"red",
             "Reactive_Microglia": "red",
             "Resting_Microglia": "magenta" } 
#%%
# generate cell type
tissue_shape = [cellType_df.centroid_x.max(),cellType_df.centroid_y.max()]  # [49849, 31824]
#df["centroid_y"] = tissue_shape[1] - df["centroid_y"] 
    #%% Cropt left and right 

rightCrop_range = [33179, 856,33179+6522, 6943]  # xmin,ymin,xmax,ymax
leftCrop_range =  [10148, 856,10148+6522, 6943]  # xmin,ymin,xmax,ymax

Crop_range = rightCrop_range 
cellType_df["rightCrop"] = (  np.array(cellType_df.centroid_x >Crop_range[0]) *
                              np.array(cellType_df.centroid_x <Crop_range[2]) *                    
                              np.array(cellType_df.centroid_y >Crop_range[1]) * 
                              np.array(cellType_df.centroid_y <Crop_range[3]) ).astype(np.bool)

Crop_range = leftCrop_range
cellType_df["leftCrop"] = (  np.array(cellType_df.centroid_x  >Crop_range[0]) * 
                             np.array(cellType_df.centroid_x <Crop_range[2]) *                     
                             np.array(cellType_df.centroid_y >Crop_range[1]) *
                             np.array(cellType_df.centroid_y <Crop_range[3]) ).astype(np.bool)

cellType_df["side"] = cellType_df["leftCrop"]*1 +cellType_df["rightCrop"]*2
cellType_df.centroid_y = cellType_df.centroid_y.max() - cellType_df.centroid_y  # filp y for visualization

#%%
INDATA = cellType_df.query("side>0")
plt.figure(dpi=200)
plt.scatter(cellType_df.centroid_x , cellType_df.centroid_y, s=1)
plt.scatter(INDATA.centroid_x , INDATA.centroid_y, s=1)
plt.savefig("loc.png")

#%%
cellType_cl = {
             "GABAergic_Neuron":"green",
             "NonGABAergic_Neuron":"blue",                                 
             "Reactive_Astrocyte": "orange",
             "Resting_Astrocyte": "yellow",                         
             "Reactive_Microglia": "red",
             "Resting_Microglia": "magenta" } 
plt.figure(dpi=200)
plt.style.use('dark_background')
for cellType in cellType_cl.keys():
    plt.scatter(cellType_df.centroid_x[cellType_df[cellType]==1] , 
                cellType_df.centroid_y[cellType_df[cellType]==1], 0.05, c = cellType_cl[cellType])

#%%
# generate neighborhood table 
# compute disctance pairwisely



CenterNames = ["GABAergic_Neuron","NonGABAergic_Neuron"]
neighborNames =[ "Reactive_Astrocyte","Resting_Astrocyte", "Reactive_Microglia","Resting_Microglia"]

attractionTable = None
for c_i, center_type in enumerate(CenterNames):    
    for side in [1,2]:
        Ptable = cellType_df[cellType_df.side == side]
        attractionTable_new, attraction_index =  utils.attractionDefine (Ptable,
                                                                     center_type = center_type,
                                                                     neighbor_type_ls =neighborNames,
                                                                     RMax = 200,
                                                                     index_opt = "kde")      
        neighborhood_pairs = [x + "IN"+ center_type  for x in attractionTable_new.columns]
        attractionTable_new.columns = neighborhood_pairs
        attractionTable_new["side"] = side
        
        if attractionTable is None:
            attractionTable = attractionTable_new  
        else:
            attractionTable = attractionTable.append(attractionTable_new)
#            
#%%        
attractionTable =attractionTable.fillna(0)


attractionTable.to_csv(os.path.abspath("../data/Storke_NeighborhoodAttraction_table.csv"))


#%%
data = np.array( attractionTable[attractionTable.columns.difference(["side"])] ) 
label = cellType_df["side"][attractionTable.index]
w_Lasso, w_treeLasso = runFeatureSelectionCompare (data = data,
                            label = label,
                            CenterName= CenterNames,
                            NeighborName = neighborNames)

#%%
plt.figure() 

df = pd.DataFrame({'Neighborhood':attractionTable.columns.difference(["side"]),
                    'Neighborhood Attraction':w_treeLasso})
ax = df.plot.barh(x='Neighborhood', y='Neighborhood Attraction')
plt.legend([])
plt.tight_layout()
#%%
# biggest changes
#obvious_pair_id =  np.argmax(w_treeLasso) 

for obvious_pair_id in range(len(w_treeLasso))[:1]:
    obvious_pair_name = attractionTable.columns[obvious_pair_id]
    obvious_attractionTable = attractionTable[obvious_pair_name]
    index_mean_df = attractionTable.groupby(["side"]).mean()
    
    c_type = obvious_pair_name.split("IN")[1]
    n_type = obvious_pair_name.split("IN")[0]
    
    fig = utils.scatter_neighbor_plot(cellType_df,center_type = c_type,  neighbor_type = n_type,
                                           tissue_ls =["leftCrop", "rightCrop"] ,
                                           title_ls = ["{:.2f}".format( index_mean_df[obvious_pair_name][1] ) ,
                                                       "{:.2f}".format( index_mean_df[obvious_pair_name][2] ) ],
                                            c_color = cellType_cl[c_type],   n_color =cellType_cl[n_type]
                                         )
    plt.suptitle( "LASSO Coeff="+ "{:.2f}".format( w_treeLasso[obvious_pair_id]))
    fig.savefig("Plot Vis"+obvious_pair_name+".png" )
    plt.close()

