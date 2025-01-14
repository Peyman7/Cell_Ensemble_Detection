# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:52:46 2022

@author: p.nazarirobati
"""
import numpy as np
import pandas as pd
import scipy as sci
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance
import math

def assembly_assignment_matrix(As_across_bins,nneu,BinSizes,display):
    

    #nAss_final=100
    nAss_final = len(As_across_bins)
    
    AAT = np.empty((nneu, nAss_final))
    AAT[:]=np.nan
    Binvector = [[]] * nAss_final

    for i in range(nAss_final):
        AAT[As_across_bins[i]['neurons'], i] = As_across_bins[i]['lags']/As_across_bins[i]['bin_size']
        Binvector[i] = As_across_bins[i]['bin_size']

    if display == 'raws':
        Unit_order = range(nneu)
        As_order = range(len(As_across_bins))

    if display =='ordunit':
        aus = np.all(np.isnan(AAT),1)
        idx_nan = np.where(aus==1)
        idx_activ = np.where(aus==0)
        AAT = np.array([[AAT[np.sum(np.isnan(AAT),axis=1)]], [AAT[np.sum(np.isnan(AAT),axis=1)]]])
        print(np.shape(AAT))
        Unit_order = [idx_activ, idx_nan]
        As_order = range(len(As_across_bins))

    if display == 'clustered':
        aus = np.all(np.isnan(AAT), axis=1)
        idx_nan = np.where(aus==1)
        idx_activ = np.where(aus==0)
        aus = np.all(np.isnan(AAT), axis=1)
        AAT_activ = AAT[aus,:]

        ## order on the base of units co-occurrence 
        A01 = np.isnan(AAT_activ)
        M_assemb = np.zeros((AAT_activ,1))

        for n in range(np.shape(A01,1)):
            aus = np.where(A01[:,n]==1)
            M_assemb[aus, aus] = M_assemb[aus,aus]+1
        M_assemb[np.where(M_assemb==0)] = 0.0001
        d_assemb = np.ones((AAT_activ,1)) / M_assemb
        d_assemb = d_assemb - np.diag(np.diag(d_assemb))
        Q = hierarchy.linkage(d_assemb,'average')
        [perm] = hierarchy.dendrogram(Q,0)
        perm1 = idx_activ[perm]

        ## order on the base of assemblies distance assemblies

        D = distance.pdist(np.transpose(np.isnan(AAT_activ)))
        Z = distance.squareform(D)
        Q2 = hierarchy.linkage(Z, 'average')
        [perm2] = hierarchy.dendrogram(Q2,0)

        AAT = AAT_activ[perm, perm2]
        AAT = [AAT, np.nan((len(idx_nan), np.shape(AAT,1)))]
        Unit_order = [perm1, idx_nan]
        Binvector = [Binvector[xx] for xx in perm2]
        As_order = perm2
  
    Amatrix = AAT
    binmat = np.reshape(np.array([Binvector[tt]/10 for tt in range(len(Binvector))]), (1, len(Binvector)))

    ### Plot Results
    #assignment matrix
    
    plt.figure(figsize=[8,6])
    ax0 = plt.subplot(2,1,1)
    h=ax0.pcolormesh(AAT, cmap='jet', vmin=np.nanmin(AAT), vmax=np.nanmax(AAT), edgecolors=[0.85,0.85,0.85], linewidth=0.004)
    # assign x,y Ticks
    ax0.set_yticks(np.arange(np.shape(AAT)[0]), minor=False)
    ax0.set_xticks(np.arange(np.shape(AAT)[1]) + 0.5, minor=False)
    ### assign x,y Ticks labels
    column_labels = list(range(1,np.shape(AAT)[1]+1))
    row_labels = list(range(np.shape(AAT)[0]))
    ax0.set_xticklabels(column_labels, minor=False, fontsize=4)
    ax0.set_yticklabels(row_labels, minor=False, fontsize=8)
    ### Y axis label 
    ax0.set_ylabel('Unit #')
    cax = ax0.inset_axes([1.02, 0.0, 0.019, 1], transform=ax0.transAxes)   # colorbar
    cbar = plt.colorbar(h, orientation="vertical", ax=ax0, cax=cax, aspect=0.5)
    cbar.set_label('Time lag / (# bins)', rotation=90)
   
    # Temporal precision colormap
    ax1 = plt.subplot(2,1,2)
    BinSizes = [xx/10 for xx in BinSizes]
    if (len(BinSizes)>1):
        vmin2 = min(BinSizes)
        vmax2 = max(BinSizes)
        print(vmin2, vmax2)
    else:
        vmin2 = math.log10(min(BinSizes-0.001))
        vmax2 = math.log10(max(BinSizes+0.001))
        print(vmin2, vmax2)
    #L=[0.001,0.002,0.003,0.004,0.005, 0.006,0.007,0.008,0.009,0.01,0.02,0.03,
    #0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
    #1,20,30,40,50,60,70,80,90,1000]
    
    L=[0.01,1,3,5,10,25,35,50,65,75,90,100]
    
    #L=[0.1, 0.5, 0.6, 0.8, 0.9, 1, 1.002,1.005, 1.008, 1.01,1.02,1.05,1.08,1.09, 1.12,1.16,1.18,1.2, 1.23, 1.25, 1.5,2,
    #2.2]
    #L = [10**3, 10**5, 10**10, 10**25, 10**35, 10**50, 10**65, 10**75, 10**90, 10**100]
    v = ax1.pcolormesh(binmat, cmap='jet', vmin=vmin2, vmax=vmax2, edgecolors=[0,0,0], linewidth=0.04)
    column_labels2 = list(np.arange(0, np.shape(AAT)[1],2))
    ax1.set_xticks(np.arange(0, np.shape(AAT)[1]+0.5,2), minor=False)
    #ax1.set_xticklabels(column_labels2, minor=False)

    cax2 = ax1.inset_axes([0.0, -5.52, 1, 0.4], transform=ax1.transAxes)

    hC = plt.colorbar(v, orientation="horizontal", ax=ax1, cax=cax2, ticks=[round(xx,0) for xx in L], aspect=0.5)
    hC.set_label('Temporal Precision \u0394 (ms)', rotation=0)

    plt.xlabel('Assembly #')

    ax1.set_aspect(1.5)
    #plt.subplots_adjust(bottom=1, hspace = 0.001)
    plt.show()

    return Amatrix,Binvector, Unit_order, As_order