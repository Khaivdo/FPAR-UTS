#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:53:40 2019

@author: james
"""

import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
print (input)
target = torch.empty(3, dtype=torch.long).random_(5)
print (target)
output = loss(input, target)
print (output)
output.backward()