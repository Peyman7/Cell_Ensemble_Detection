# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:05:28 2022

@author: p.nazarirobati
"""
import pickle
import os

def assemblies_data(path_name):
    assemblies = []
    for root, dirs, files  in os.walk(path_name):
        for file in sorted(files):
            print(file)
            if file.endswith('.pkl'):
                with open(os.path.join(root, file), 'rb') as f:
                    patt = pickle.load(f)
                    assemblies.append(patt)
    return assemblies