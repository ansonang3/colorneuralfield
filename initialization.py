#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:52:59 2018

Last modified: Fri Dec 14 2018

@author: Anna SONG

Here we give some examples of (nearly) optimal values for the vector of parameters q, which can also be used as
initialization values.

"""
import numpy as np

# These optimal parameters correspond to:
# observers `MC', `AZ' from [Monnier & Shevell (2004)]
# a mean observer for data from [Monnier (2008)]
# to observer `AS' for our HSL data

q_MC =  np.array([ 0.60079587,  0.68869341,  0.3012414 ,  0.39985751,  4.42097036,
        1.81886753,  0.58379431,  8.35000022,  0.47      ,  0.3       ,
        1.8       ])# optimized for MC_mean_2

q_AZ = np.array([ 0.60396227,  0.69208158,  0.30630672,  0.40419103,  4.42373092,
        1.81448358,  0.60060304,  8.34999534,  0.47      ,  0.3       ,
        1.8       ]) # optimized for AZ_mean_2

q_nonlin = np.array([ 0.41916268,  0.70936422,  0.62977593,  1.1601163 ,  4.42988436,
        1.71983088,  0.56132954,  6.34999714,  0.47      ,  0.3       ,
        1.8       ]) # optimized for nonlinear data of 2008

q_HSL = np.array([ 0.73143291,  0.15000446,  0.52468157,  0.67628293,  4.41133491,
        1.83599656,  0.51234975,  8.350026  ,  0.47      ,  0.3       ,
        1.8       ]) # optimized for our data in 2D HSL


# These optimal parameters correspond to artificial data

q_Alice = np.array([0.6031818 , 0.6864943 , 0.30439608, 0.40044069, 4.42225434,
       1.8156906 , 0.59823715, 8.349993  , 0.47      , 0.3       ,
       1.8       ]) # optimized for Alice

q_Bob = np.array([0.59890043, 0.68390427, 0.29681706, 0.39549819, 4.419717, 1.8194687,
                   0.58398932, 8.34999625, 0.47,0.3, 1.8]) # optimized for Bob