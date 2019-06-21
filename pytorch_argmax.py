#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:19:32 2018

Last modified: Fri Dec 14 2018

@author: Anna SONG

Please see the related article for a global description of our algorithms.

For further information on pytorch and its usage, please refer to:
    [A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito,
    Z. Lin, A. Desmaison, L. Antiga, and A. Lerer, Automatic differentiation in pytorch,
    in NIPS-W, 2017]
    
    
Companion script for:
    - main_pytorch.py
    - main_HSL_pytorch_2D.py
to compute soft argmax and soft max (L infinity norms).
    

"""
import torch
import torch.nn as nn
from torch.autograd import Variable

dtype = torch.FloatTensor

from settings import color_abs
color_abs_var = Variable(torch.from_numpy(color_abs).type(dtype))

gam = Variable(torch.FloatTensor([10]))

class SoftArgmax(nn.Module):
    ''' to compute a soft argmaximum in the color space defined by color_abs '''
    def __init__(self):
        super(SoftArgmax,self).__init__()
    
    def forward(self,v):
        for i in range(2) : # 2
            # as many times as necessary to clearly separate too similar max peaks
            v = v/v.max()
            v = torch.exp(10*v)
        if len(v.shape) == 2 : # shape NN * Ncol
            v = v/((torch.sum(v,dim = 1))[:,None])
            argmaxs = torch.sum(color_abs_var*v,dim = 1)
            return argmaxs # shape (NN,)
        if len(v.shape) == 1 :
            v = v/v.sum()
            argmax = torch.sum(color_abs_var*v)
            return argmax # scalar
        
class SoftMax(nn.Module):
    ''' to compute a soft infinite norm '''
    def __init__(self):
        super(SoftMax,self).__init__()
    
    def forward(self,v):
        if len(v.shape) == 2 : # shape NN * Ncol and between -2 and 2
            v = torch.exp(20*(v-2)) # to have negative numbers in the exp
            maxs = torch.log(torch.sum(v,1))/20 + 2
            return maxs # shape (NN,)
        if len(v.shape) == 1 :
            v = torch.exp(20*(v-2))
            maxi = torch.log(v.sum(0))/20 + 2
            return maxi
        if len(v.shape) == 3 : # shape Nexp * (NN = Ncol) * Ncol and between -2 and 2
            v = torch.exp(20*(v-2)) # to have negative numbers in the exp
            maxs = torch.log(torch.sum(v,2))/20 + 2
            return maxs # shape (Nexp,NN)
 
class HSL_SoftArgmax_2D(nn.Module):
    ''' to compute a soft argmaximum in the HSL (chromatic 2D) color space
    defined by some argument here called abs_prod (or color_abs_prod)'''
    def __init__(self):
        super(HSL_SoftArgmax_2D,self).__init__()

    def forward(self,v,abs_prod): # v shape (Nexp, NN = Ncol)
        C1 = Variable(torch.from_numpy(abs_prod[:,0]).type(dtype))#.view(-1)
        C2 = Variable(torch.from_numpy(abs_prod[:,1]).type(dtype))#.view(-1)
        for i in range(2) :
            """ as many times as necessary to clearly separate too similar max peaks """
            v = v/v.max(1)[0][:,None]
            v = torch.exp(10*v)
        v = v/(v.sum(1)[:,None])
        v1 = (C1[None]*v).sum(1)
        v2 = (C2[None]*v).sum(1)
        return torch.t(torch.cat((v1[None],v2[None]))) # shape (Nexp,2)

