# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 13:29:43 2022

@author: p.nazarirobati
"""

import numpy as np
import pandas as pd

def epochs_duration(t_rem1, t_rem2, t_sws1, t_sws2, t_lvs1, t_lvs2, t_lvsk1, t_lvsk2, t_spw1, t_spw2, t_sleep1, t_sleep2, t_tks, t_tke):
    
    ep_duration = pd.DataFrame(columns=['tm_rem1', 'tm_rem2', 'tm_sws1', 'tm_sws2', 'tm_spw1', 'tm_spw2', 'tm_lvs1', 'tm_lvs2','tm_lvsk1', 'tm_lvsk2', 'tm_lvsspw1', 'tm_lvsspw2', 'tm_lvskspw1', 'tm_lvskspw2', 'tm_s1', 'tm_s2', 'tm_tsk'])
    tm_rem1 = 0
    tm_rem2=0
    tm_sws1=0
    tm_sws2=0
    tm_lvs1=0
    tm_lvs2=0
    tm_lvsk1=0
    tm_lvsk2=0
    tm_spw1=0
    tm_spw2=0
    tm_lvsspw1=0
    tm_lvsspw2=0
    tm_lvskspw1=0
    tm_lvskspw2=0
    tm_sleep1=0
    tm_sleep2=0
    k=0
    
    for i in range(np.shape(t_rem1)[0]):
        tm = t_rem1[i,1] - t_rem1[i,0]
        tm_rem1 = tm_rem1 + tm
    
    for i in range(np.shape(t_rem2)[0]):
        tm = t_rem2[i,1] - t_rem2[i,0]
        tm_rem2 = tm_rem2 + tm
        
    for i in range(np.shape(t_sws1)[0]):
        tm = t_sws1[i,1] - t_sws1[i,0]
        tm_sws1 = tm_sws1 + tm
    
    for i in range(np.shape(t_sws2)[0]):
        tm = t_sws2[i,1] - t_sws2[i,0]
        tm_sws2 = tm_sws2 + tm
    
    for i in range(np.shape(t_lvs1)[0]):
        tm = t_lvs1[i,1] - t_lvs1[i,0]
        tm_lvs1 = tm_lvs1 + tm
        
    for i in range(np.shape(t_lvs2)[0]):
        tm = t_lvs2[i,1] - t_lvs2[i,0]
        tm_lvs2 = tm_lvs2 + tm
    
    for i in range(np.shape(t_lvsk1)[0]):
        tm = t_lvsk1[i,1] - t_lvsk1[i,0]
        tm_lvsk1 = tm_lvsk1 + tm
    
    for i in range(np.shape(t_lvsk2)[0]):
        tm = t_lvsk2[i,1] - t_lvsk2[i,0]
        tm_lvsk2 = tm_lvsk2 + tm
        
    for i in range(np.shape(t_spw1)[0]):
        tm = t_spw1[i,1] - t_spw1[i,0]
        tm_spw1 = tm_spw1 + tm
    
    for i in range(np.shape(t_spw2)[0]):
        tm = t_spw2[i,1] - t_spw2[i,0]
        tm_spw2 = tm_spw2 + tm
        
    for i in range(np.shape(t_lvs1)[0]):
        for j in range(np.shape(t_spw1)[0]):
            if (t_spw1[j,0]<=t_lvs1[i,0] and t_spw1[j,1]>=t_lvs1[i,0]) or (t_spw1[j,1]>=t_lvs1[i,1] and t_spw1[j,0]<t_lvs1[i,1]) or (t_spw1[j,0]>=t_lvs1[i,0] and t_spw1[j,1]<=t_lvs1[i,1]):
                tm = min(t_spw1[j,1],t_lvs1[i,1]) - max(t_spw1[j,0], t_lvs1[i,0])
                if tm<0:
                    print(i)
                    print(j)
                tm_lvsspw1 = tm_lvsspw1 + tm
                break
    
    for i in range(np.shape(t_lvs2)[0]):
        for j in range(np.shape(t_spw2)[0]):
            if (t_spw2[j,0]<=t_lvs2[i,0] and t_spw2[j,1]>=t_lvs2[i,0]) or (t_spw2[j,1]>=t_lvs2[i,1] and t_spw2[j,0]<t_lvs2[i,1]) or (t_spw2[j,0]>=t_lvs2[i,0] and t_spw2[j,1]<=t_lvs2[i,1]):
                tm = min(t_spw2[j,1],t_lvs2[i,1]) - max(t_spw2[j,0], t_lvs2[i,0])
                if tm<0:
                    print(i)
                    print(j)
                tm_lvsspw2 = tm_lvsspw2 + tm
                break
    
    for i in range(np.shape(t_lvsk1)[0]):
        for j in range(np.shape(t_spw1)[0]):
            if (t_spw1[j,0]<=t_lvsk1[i,0] and t_spw1[j,1]>=t_lvsk1[i,0]) or (t_spw1[j,1]>=t_lvsk1[i,1] and t_spw1[j,0]<t_lvsk1[i,1]) or (t_spw1[j,0]>=t_lvsk1[i,0] and t_spw1[j,1]<=t_lvsk1[i,1]):
                tm = min(t_spw1[j,1],t_lvsk1[i,1]) - max(t_spw1[j,0], t_lvsk1[i,0])
                if tm<0:
                    print(i)
                    print(j)
                tm_lvskspw1 = tm_lvskspw1 + tm
                break
    
    for i in range(np.shape(t_lvsk2)[0]):
        for j in range(np.shape(t_spw2)[0]):
            if (t_spw2[j,0]<=t_lvsk2[i,0] and t_spw2[j,1]>=t_lvsk2[i,0]) or (t_spw2[j,1]>=t_lvsk2[i,1] and t_spw2[j,0]<t_lvsk2[i,1]) or (t_spw2[j,0]>=t_lvsk2[i,0] and t_spw2[j,1]<=t_lvsk2[i,1]):
                tm = min(t_spw2[j,1],t_lvsk2[i,1]) - max(t_spw2[j,0], t_lvsk2[i,0])
                if tm<0:
                    print(i)
                    print(j)
                tm_lvskspw2 = tm_lvskspw2 + tm
                break
    
    for i in range(np.shape(t_sleep1)[0]):
        tm = t_sleep1[i,1] - t_sleep1[i,0]
        tm_sleep1 = tm_sleep1 + tm
        
    for i in range(np.shape(t_sleep2)[0]):
        tm = t_sleep2[i,1] - t_sleep2[i,0]
        tm_sleep2 = tm_sleep2 + tm
     
    tm_tsk = t_tke - t_tks
    
    
    
    
    ep_duration.loc[0,'tm_rem1']=tm_rem1 
    ep_duration.loc[0,'tm_rem2']=tm_rem2
    ep_duration.loc[0,'tm_sws1']=tm_sws1 
    ep_duration.loc[0,'tm_sws2']=tm_sws2
    ep_duration.loc[0,'tm_spw1']=tm_spw1
    ep_duration.loc[0,'tm_spw2']=tm_spw2
    ep_duration.loc[0,'tm_lvs1']=tm_lvs1 
    ep_duration.loc[0,'tm_lvs2']=tm_lvs2 
    ep_duration.loc[0,'tm_lvsk1']=tm_lvsk1 
    ep_duration.loc[0,'tm_lvsk2']=tm_lvsk2 
    ep_duration.loc[0,'tm_lvsspw1']=tm_lvsspw1 
    ep_duration.loc[0,'tm_lvsspw2']=tm_lvsspw2 
    ep_duration.loc[0,'tm_lvskspw1']=tm_lvskspw1 
    ep_duration.loc[0,'tm_lvskspw2']=tm_lvskspw2 
    ep_duration.loc[0, 'tm_s1'] = tm_sleep1
    ep_duration.loc[0, 'tm_s2'] = tm_sleep2
    ep_duration.loc[0, 'tm_tsk'] = tm_tsk

    
    return ep_duration
    
        
    