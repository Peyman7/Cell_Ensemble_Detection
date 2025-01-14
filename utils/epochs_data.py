# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:01:20 2022

@author: p.nazarirobati
"""
import numpy as np
import os
import pandas as pd

def epochs_data(path_name):
    
    
    dict_epochs = {}
    for root, dirs, files  in os.walk(path_name):
        for file in files:

            if file.startswith('epochs.session'):   ### recording start and end time
                with open(os.path.join(root, file), 'r') as f:
                    session = np.reshape(np.fromstring(f.read(), dtype=np.float64, sep= ' '), (1,2))
                    t_ss = float(session[0,0])
                    t_es = float(session[0,1])
            if file.startswith('epochs.sleep1'):   # sleep 1 
                with open(os.path.join(root, file), 'r') as f1:
                    s1 = np.reshape(np.fromstring(f1.read(), dtype=np.float64, sep= ' '), (1,2))
                    t_s1s = float(s1[0,0])
                    t_s1e = float(s1[0,1])
            if file.startswith('epochs.sleep2'):   # sleep 2
                with open(os.path.join(root, file), 'r') as f2:
                    s2 = np.reshape(np.fromstring(f2.read(), dtype=np.float64, sep= ' '), (1,2))
                    t_s2s = float(s2[0,0])
                    t_s2e = float(s2[0,1])  
            if file.startswith('epochs.task'):    # task
                with open(os.path.join(root, file), 'r') as f3:
                    tsk = np.reshape(np.fromstring(f3.read(), dtype=np.float64, sep= ' '), (1,2))
                    t_tks = float(tsk[0,0])
                    t_tke = float(tsk[0,1])
             
            if file.startswith('epochs.tsrem1'):    # task
                with open(os.path.join(root, file), 'r') as f3:
                    t_rem1 = np.fromstring(f3.read(), dtype=np.float64, sep= ' ')
                    t_rem1 = np.reshape(t_rem1, (int(len(t_rem1)/2), 2))
                    
            if file.startswith('epochs.tsrem2'):    # task
                with open(os.path.join(root, file), 'r') as f4:
                    t_rem2 = np.fromstring(f4.read(), dtype=np.float64, sep= ' ')
                    t_rem2 = np.reshape(t_rem2, (int(len(t_rem2)/2), 2))
            
            if file.startswith('epochs.tssws1'):    # task
                with open(os.path.join(root, file), 'r') as f5:
                    t_sws1 = np.fromstring(f5.read(), dtype=np.float64, sep= ' ')
                    t_sws1 = np.reshape(t_sws1, (int(len(t_sws1)/2), 2))
           
            if file.startswith('epochs.tssws2'):    # task
                with open(os.path.join(root, file), 'r') as f6:
                    t_sws2 = np.fromstring(f6.read(), dtype=np.float64, sep= ' ')
                    t_sws2 = np.reshape(t_sws2, (int(len(t_sws2)/2), 2))
            
            if file.startswith('epochs.tsspw1'):    # task
                with open(os.path.join(root, file), 'r') as f7:
                    t_spw1 = np.fromstring(f7.read(), dtype=np.float64, sep= ' ')
                    t_spw1 = np.reshape(t_spw1, (int(len(t_spw1)/2), 2))
            
            if file.startswith('epochs.tsspw2'):    # task
                with open(os.path.join(root, file), 'r') as f8:
                    t_spw2 = np.fromstring(f8.read(), dtype=np.float64, sep= ' ')
                    t_spw2 = np.reshape(t_spw2, (int(len(t_spw2)/2), 2))
            
            if file.startswith('epochs.tslvs1'):    # task
                with open(os.path.join(root, file), 'r') as f9:
                    t_lvs1 = np.fromstring(f9.read(), dtype=np.float64, sep= ' ')
                    t_lvs1 = np.reshape(t_lvs1, (int(len(t_lvs1)/2), 2))
            
            if file.startswith('epochs.tslvs2'):    # task
                with open(os.path.join(root, file), 'r') as f10:
                    t_lvs2 = np.fromstring(f10.read(), dtype=np.float64, sep= ' ')
                    t_lvs2 = np.reshape(t_lvs2, (int(len(t_lvs2)/2), 2))
            
            if file.startswith('epochs.tslvsk1'):    # task
                with open(os.path.join(root, file), 'r') as f11:
                    t_lvsk1 = np.fromstring(f11.read(), dtype=np.float64, sep= ' ')
                    t_lvsk1 = np.reshape(t_lvsk1, (int(len(t_lvsk1)/2), 2))
            
            if file.startswith('epochs.tslvsk2'):    # task
                with open(os.path.join(root, file), 'r') as f12:
                    t_lvsk2 = np.fromstring(f12.read(), dtype=np.float64, sep= ' ')
                    t_lvsk2 = np.reshape(t_lvsk2, (int(len(t_lvsk2)/2), 2))
                                       
                    
            if file.startswith('epochs.tssleep1'):    # task
                with open(os.path.join(root, file), 'r') as f13:
                    t_sleep1 = np.fromstring(f13.read(), dtype=np.float64, sep= ' ')
                    t_sleep1 = np.reshape(t_sleep1, (int(len(t_sleep1)/2), 2))

            if file.startswith('epochs.tssleep2'):    # task
                with open(os.path.join(root, file), 'r') as f14:
                    t_sleep2 = np.fromstring(f14.read(), dtype=np.float64, sep= ' ')
                    t_sleep2 = np.reshape(t_sleep2, (int(len(t_sleep2)/2), 2))
                    
            if file.startswith('epochs.tsk1'):    # task
                with open(os.path.join(root, file), 'r') as f15:
                    t_so1 = pd.read_csv(f15, delimiter = "\s", header=None)
                    t_so1 = t_so1.to_numpy()

            if file.startswith('epochs.tsk2'):    # task
               with open(os.path.join(root, file), 'r') as f16:
                   t_so2 = pd.read_csv(f16, delimiter = "\s", header=None)
                   t_so2 = t_so2.to_numpy()
                   
            if file.startswith('epochs.tswake1'):    # task
                with open(os.path.join(root, file), 'r') as f17:
                    t_wake1 = pd.read_csv(f17, delimiter = "\s", header=None)
                    t_wake1 = t_wake1.to_numpy()

            if file.startswith('epochs.tswake2'):    # task
               with open(os.path.join(root, file), 'r') as f18:
                   t_wake2 = pd.read_csv(f18, delimiter = "\s", header=None)
                   t_wake2 = t_wake2.to_numpy()
                   
    dict_epochs['session'] = [t_ss, t_es]
    dict_epochs['sleep1'] = [t_s1s, t_s1e]  
    dict_epochs['sleep2'] = [t_s2s, t_s2e] 
    dict_epochs['task'] = [t_tks, t_tke]
    dict_epochs['t_rem1'] = t_rem1
    dict_epochs['t_rem2'] = t_rem2
    dict_epochs['t_sws1'] = t_sws1
    dict_epochs['t_sws2'] = t_sws2
    dict_epochs['t_spw1'] = t_spw1
    dict_epochs['t_spw2'] = t_spw2
    dict_epochs['t_lvs1'] = t_lvs1
    dict_epochs['t_lvs2'] = t_lvs2
    dict_epochs['t_lvsk1'] = t_lvsk1
    dict_epochs['t_lvsk2'] = t_lvsk2
    #dict_epochs['t_so1'] = t_so1
    #dict_epochs['t_so2'] = t_so2
    dict_epochs['t_sleep1'] = t_sleep1
    dict_epochs['t_sleep2'] = t_sleep2
    dict_epochs['t_wake1'] = t_wake1
    dict_epochs['t_wake2'] = t_wake2





               
    return dict_epochs  
        
    #return t_ss, t_es, t_s1s, t_s1e, t_s2s, t_s2e, t_tks, t_tke, t_rem1, t_rem2, t_sws1, t_sws2, t_spw1, t_spw2, t_lvs1, t_lvs2, t_lvsk1, t_lvsk2, t_sleep1, t_sleep2, t_tsk1, t_tsk2  

