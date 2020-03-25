u# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:52:09 2019

@author: xli63
"""


'''
Denstiy-weighted Colocation rule finding 

Author: Rebecca LI, University of Houston, Farsight Lab, 2019
xiaoyang.rebecca.li@gmail.com

-- Reference --
[1]Yao, Xiaojing, et al. "A co-location pattern-mining algorithm with a density-weighted distance 
thresholding consideration." Information Sciences 396 (2017): 144-161.
[2]X. Yao, L. Peng, L Yang, T. Chi, A fast space-saving algorithm for maximal co-location pattern 
mining, Expert Syst. Appl. 63 (2016) 310323.

-- Function --
Input:
    Instances' type and location using X and Y coordinates. 
Parameters:
    The distance threshold, r, and the prevalence threshold, Min-preKDE, 
'''


import sys, os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
#     # Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd()+'\Libfcts')
import utils 

def associationRule(candidateSet, supp_thres):
    #supp_thres: the minimum threshold
    #Apriori principle:
    #If an itemset is frequent, then all of its subsets must also be frequent

    frequencySet={}
    for neiName in candidateSet.columns.tolist():
        if "-" not in neiName:                    # neiName only contains one element
            if (candidateSet[neiName] >= supp_thres)[0]:
                frequencySet[neiName] = candidateSet[neiName]
        else:                                     # neiName contains more than one elements
            nei_cliques = utils.instance_cliques(neiName.split("-")  ) 
            if np.array( [(candidateSet['-'.join(nc) ] >= supp_thres)[0]for nc in nei_cliques]).all():
                frequencySet[neiName] = candidateSet[neiName]
                
    frequencySet =    pd.DataFrame.from_dict(frequencySet, orient = 'index')           
    return frequencySet    

def str2bool(str_input):
    bool_result = True if str_input.lower() in ["t", 'true', '1', "yes", 'y'] else False
    return bool_result

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import argparse
    import time

    parser = argparse.ArgumentParser(description='*** Denstiy-weighted Colocation rule finding ',
                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-d', '--demo', default='F', type=str,
                        help=" 'T' only match channel 0, 'F' match all channels")
    parser.add_argument('-i', '--input_dir', 
                        default=r"/brazos/roysam/xli63/exps/Data/50_plex/jj_final/multiclass/icePhenotyping/classification_table.csv",
                        help='Path to the directory of input images')
    parser.add_argument('-o', '--output_dir',
                        default=r"/brazos/roysam/xli63/exps/Data/50_plex/jj_final/multiclass/icePhenotyping/Colocations_ParaCompare.csv",
                        help='Path to the directory to save output images')
    args = parser.parse_args()

    start = time.time()

    if str2bool(args.demo) == False:
        print ("Run full")
        Ptable=pd.read_csv(args.input_dir)
    else:
        print ("Run demo")
        Ptable=pd.read_csv(args.input_dir)[0:100]

    ## Setting
    center_type=['NeuN']  # only visit 1 
    neighbor_type_ls=['Iba1', 'Olig2', 'RECA1', 'S100'] # visit >=1
    supp_thres=0.05                 # only to do frequency set
    # Rthres_ls = range(1,200) # vicinity 
    Rthres_ls = range(10,550,10) # far neighborhood
    index_opt_ls= ["normal","kde"]

    for index_opt in index_opt_ls:
        index_opt_start = time.time()
        ParaCompare={}
        for Rthres in Rthres_ls:
    #         print ("...",index_opt, Rthres)
            attractionTable, attraction_index =   utils.attractionDefine (Ptable,center_type,neighbor_type_ls, Rthres,index_opt)  

            ParaCompare[index_opt+"_Rmax"+str(Rthres)] = attraction_index.values[0]        
            # calculate out the association rules 
            
            # frequencySet = associationRule(attraction_index,supp_thres)  # select the significant items, unselected ones are NAN

        ParaCompare=pd.DataFrame.from_dict(ParaCompare).set_index(attraction_index.columns.values).transpose()
        ParaCompare.to_csv(os.path.join(args.output_dir, "Far_attraction_index_"+ index_opt +".csv"))        
        print(index_opt ,' finished successfully in {} seconds.'.format( time.time() - index_opt_start))

        # visualization 
        plt.figure(figsize=(10,6))
        for neiName in ParaCompare.columns.tolist():
            if "-" not in neiName:
                marker = "-"
            else:
                if len( neiName.split("-") ) == 1:
                    marker = "."
                elif len(neiName.split("-")) == 2:
                    marker = "--"
                elif len(neiName.split("-") )== 3:
                    marker = "-."
                elif len(neiName.split("-") )== 4:
                    marker = ":"
            plt.plot(np.array(Rthres_ls),np.array(ParaCompare[neiName]),marker)

        plt.legend(ParaCompare.columns.tolist(),loc ='upper left')
        plt.title("Neighborhood Attraction Index of all instance cliques " + index_opt)
        plt.savefig(os.path.join(args.output_dir, "Far_attraction_index_"+ index_opt +".tif"))

        # plt.show()

    print('*' * 50)
    print ("Save in ",args.output_dir)
    print('*' * 50)
    print('Pipeline finished successfully in {} seconds.'.format( time.time() - start))





