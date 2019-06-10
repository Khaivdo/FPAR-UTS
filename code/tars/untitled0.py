#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:15:56 2019

@author: james
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
#X = [['Male', 1], ['Female', 2], ['Female', 3]]
X = [['11_chat',0], ['12_clean', 1], ['13_drink', 2], ['14_dryer', 3], ['15_microwave',4], ['16_print', 5], 
     ['17_walk',6], ['18_shake',7]]
enc.fit(X)
b = enc.fit_transform(X).toarray()
print(b)






nb_classes = 5
data = [[2, 3, 4, 1]]

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

c = indices_to_one_hot(data, nb_classes)
print(c)