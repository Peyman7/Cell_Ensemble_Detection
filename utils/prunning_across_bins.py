# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:56:06 2022

@author: p.nazarirobati
"""
import numpy as np
import math
import pandas as pd
import scipy as sci
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from delete_multiple_element import delete_multiple_element

def prunning_across_bins(As_across_bins, As_across_bins_index, nneu, criteria, th, style):
     

    As_across_bins_pr = []
    As_across_bins_index_pr = []
    
    nA = len(As_across_bins)
    bin_vector = [[]] * len(As_across_bins)
    ATstructure = np.zeros((nneu, len(As_across_bins)))
    for i in range(len(As_across_bins)):
        ATstructure[As_across_bins[i]['neurons'],i] =1
        bin_vector[i] = As_across_bins[i]['bin_size']
    ### Prunning option: distance ####
    if criteria == 'distance':
        if (th<0) or (th>1):
            print('ERROR: cutoff can assume values inside (0,1) interval (~0 -> no clustering; ~1 -> all assemblies in the same cluster')
            return 
        #if (style != 'pvalue') or (style != 'signature'):
        #    print('ERROR: please, specify a style. style can be selected as "pvalue" or "signature."')
        #    return 
        
        D = np.zeros((nA, nA))
        a1=np.zeros((nneu, 2))
        for i in range(nA):
            for j in range(nA):
                a1[:,0] = ATstructure[:,i]
                a1[:,1] = ATstructure[:,j]
            
                D[i,j] = distance.pdist(np.transpose(a1), metric='cosine') ### it's possible to select other metrics
            D[i,i] = 0
        D = np.round(D, 4)
        df=pd.DataFrame(D)

        Z=np.round(distance.squareform(D), 4)
        #print(np.shape(Z))
        #model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='average')
        #model = model.fit(D)
        #print(model.labels_)
        #hierarchy.dendrogram(D, truncate_mode='level',p=3)
        #plt.show()
        Q2 = hierarchy.linkage(Z, method = 'average', optimal_ordering=False)

        fig = plt.figure(figsize=(10, 8))
        perm2 = hierarchy.dendrogram(Q2)
        #plt.show()
        C = hierarchy.fcluster(Q2, t=th, criterion='distance', depth=100)
        #print(C)
        #C = [11, 8, 10, 11, 10, 3, 5, 4,6,1,7,9,2,7,9,7,9,7,9,7,9]
        #C=[xx-1 for xx in C]
        
        patt = [[]]* len((np.unique(C).tolist()))
        for i in range(len(np.unique(C).tolist())):
           patt[i] = [j for j, x in enumerate(C) if x==i]
        
        #print(patt)
        L = [len(patt[xx]) for xx in range(len(patt))]
        #print(L)
        patt_multipli = [patt[i] for i in range(len(patt)) if L[i]>1]
        #print(patt_multipli)
        pat_to_remove= []
        for i in range(len(patt_multipli)):
            pr = [[]] * len(patt_multipli[i])
            Nocc=[[]] * len(patt_multipli[i])
            aus = patt_multipli[i]
            for k in range(len(patt_multipli[i])):
                A = aus[k]
                #print(A)
                pr[k] = As_across_bins[A]['pvalue']
                Nocc[k] = As_across_bins[A]['signature']
            if style =='pvalue': b = np.argmin(pr)
            if style == 'signature': b = np.argmax(Nocc)
            aus.pop(b)
            #print(b)
            pat_to_remove = np.append(pat_to_remove,aus)
        #print(pat_to_remove)
        As_across_bins_index_pr = As_across_bins_index
        As_across_bins_pr = As_across_bins
        As_across_bins_index_pr = delete_multiple_element(As_across_bins_index_pr, pat_to_remove)
        As_across_bins_pr = delete_multiple_element(As_across_bins_pr, pat_to_remove)
    
    ### Prunning option: biggest ###
    elif criteria == 'biggest':
        
        #b = [0,1,2,3,4,10,11,13,14,15,16,17,18,20,19,5,6,7,8,9,12]
        b = np.argsort(np.sum(ATstructure, axis=0))[::-1]
    
         
        As_across_bins_sorted = [As_across_bins[xx] for xx in b]
        As_across_bins_sorted_index = [As_across_bins_index[xx] for xx in b]
        bin_vector_sorted = [bin_vector[xx] for xx in b]
        
        to_remove = []
      
        for i in range(nA):
            test_elementsA = As_across_bins_sorted[i]['neurons']
            for j in range(i+1, nA):
                test_elementsB = As_across_bins_sorted[j]['neurons']

                C = [value for value in test_elementsA if value in test_elementsB]
                if len(C)==len(test_elementsA):
                    if As_across_bins_sorted[j]['pvalue'] > As_across_bins_sorted[i]['pvalue']:
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        
                elif len(C) == len(test_elementsB):
                    to_remove.append(j)
        
        to_remove_u = np.unique(np.array(to_remove))
        
        As_across_bins_sorted = delete_multiple_element(As_across_bins_sorted, to_remove_u.tolist())
        As_across_bins_sorted_index = delete_multiple_element(As_across_bins_sorted_index, to_remove_u.tolist())
        bin_vector_sorted = delete_multiple_element(bin_vector_sorted, to_remove_u.tolist())
        
        b =np.argsort(bin_vector_sorted)
        As_across_bins_index_pr = [As_across_bins_sorted_index[tt] for tt in b]
        As_across_bins_pr = [As_across_bins_sorted[tt] for tt in b]

    else:
        print('ERROR: no proper criteria enetered. ''biggest'', ''distance'' are the only acceptable criterion specifications.')
        return

    return As_across_bins_pr, As_across_bins_index_pr 