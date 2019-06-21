#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 22:18:30 2018

Last modified: Fri Dec 14 2018

@author: Anna SONG

Please see the related article for a global description of our algorithms.

settings.py fixes all the important settings for the color neural field (CNF) and its regression:
    - the parameter date [2004,2008,2017] which indicates the dataset and setting to be used, among:
        - artificial data similar to that of [Monnier, Shevell (2004)]:
            for simulating the matching experiments of some observers (PM,AZ,MC)
        - artificial data similar to that of [Monnier (2008)]:
            for reproducing the nonlinear behavior of the shifts
        - our own data: shifts in the chromatic disk of HSL, which we can also simulate.
    - various parameters, such as sampling rates for the discrete setting;
    - test_configs and comp_configs which contain the experimental values
    - functions creating cortical images from the values


Warning: the sampling resolution in physical or color space is important in order to:
    - have regular bands (otherwise, biases during computations)
    - controls the computation time: too much resolution is heavy to handle (especially for HSL setting).
"""

import numpy as np
import matplotlib.pyplot as plt

date = 2004
if date not in [2004,2008,2017]:
    raise ValueError('Please input a date in [2004,2008,2017]')

#The sampling resolution Nx is especially important and has to be checked by displaying the
# input cortical image. Check the regularity of the bands. We left default values, which are OK.

if date == 2004 : # 1D color space
    R = 1 # default value 1; delimits the domain
    LC = 2 # default value 2; Copp = [-LC, +LC]
    nb = 4
    dr = (2*R)/(2*nb+1) # band width in physical space
    dr_ann= (R-0.9/2.25)/(2*nb+1) # for displaying annuli, not used for CNF
    Nx = 10 # 10, sampling number in physical space; put Nx = 500 for displaying regular annuli
    Nc = 20 # 20, sampling number in color space
    Nexp = 6 # nb of experiments

if date == 2008 : # 1D color space
    R = 1 # default value 1; delimits the domain
    LC = 2
    nb = 8
    dr = (2*R)/(2*nb+1) # band width in physical space
    Nx = 8 # 8, sampling number in physical space
    Nc = 20 # 20, sampling number in color space
    Nexp = 7
    Nexp = 14

if date == 2017 : # 2D color space
    Nexp = 36 # at least in our dataset
    R = 1 # default value 1; delimits the domain
    LC = 1 # default value 1; Copp = [-LC, +LC]
    Nx = 4 # sampling number in physical space; too big increases computation time
    Nc = 4 # sampling number in color space; too big increases computation time

dx = R/Nx # sampling step
dc = LC/Nc # sampling step
space_abs = np.mgrid[-R:R:(2*Nx+1)*1j]
extended_abs = np.mgrid[-2*R:2*R:(4*Nx+1)*1j]
color_abs = np.mgrid[-LC:LC:(2*Nc+1)*1j]
extended_color_abs = np.mgrid[-2*LC:2*LC:(4*Nc+1)*1j]

def color(k) :
    return (k-Nc)*dc # color corresponding to index k
def color_index(c) :
    return np.round(c/dc+Nc) # index corresponding to color c
def spatial(i) :
    return (i-Nx)*dx # position corresponding to index i
def spatial_index(x) :
    return round(x/dx+Nx) # index corresponding to position x

if date in [2004,2008] :
    def create_ray_images(configs,show_C = False,dr = dr):
        Y,X = np.mgrid[0:2*Nx+1, 0:2*Nx+1]
        X_abs = spatial(X)
        if configs.ndim == 1 :
            tests,nears,farthers = configs
            C = np.zeros((2*Nx+1,2*Nx+1))
        elif configs.ndim == 2 :
            tests,nears,farthers = configs.T
            NN = len(configs)
            C = np.zeros((NN,2*Nx+1,2*Nx+1))
        elif configs.ndim == 3 :
            tests = configs[:,:,0] ; nears = configs[:,:,1] ; farthers = configs[:,:,2]
            NN = configs.shape[1]
            C = np.zeros((configs.shape[0],NN,2*Nx+1,2*Nx+1))
    
        for i in range(2*nb+1) :
            ri = R - i*dr
            'astuce de la mort qui tue pour faire des choses régulières : mettre un facteur 2'
            ray = (ri-2*dr <= X_abs)*(X_abs <= ri)
            if i == nb :
                cols = tests
            elif (i-nb)%2 == 0 :
                cols = farthers
            else :
                cols = nears
            if configs.ndim == 1 :
                C[ray] = cols[None]
            elif configs.ndim == 2 :
                C[:,ray] = cols[:,None]
            elif configs.ndim == 3 :
                C[:,:,ray] = cols[:,:,None]
            
        return C
    create_images = create_ray_images


if date == 2017 :
    def create_double_images(configs) :
        ''' after retionotopic transformation of the setting for HSL (one small square centered
        in a bigger square) and approximating, we have two vertical bands, test and near '''
        tests,nears = configs.T
        NN = len(configs)
        C = np.zeros((NN,2*Nx+1,2*Nx+1))
        C[:,:,:Nx] = tests[:,None,None]
        C[:,:,Nx:] = nears[:,None,None]
        return C
    create_images = create_double_images


if date == 2004 :
    loc = spatial_index(R - (nb+0.5)*dr) # point of interest when doing color matching
    col_diff = 0.98 # to fix the position of the neutral color (new zero)
    purple, lime, test, white = np.array([2.0,0.16,0.98,0.98])-col_diff #  some colors...
    
    Alice_test = np.array([[0.98,2.0,2.0], # pp
                       [0.98,0.16,0.16], # ll
                       [0.98,2.0,0.98], # pw
                       [0.98,0.16,0.98], # lw
                       [0.98,0.98,2.0], # wp
                       [0.98,0.98,0.16], # wl
                       [0.98,2.0,0.16], # pl --> generates the greatest shift
                       [0.98,0.16,2.0]]) # lp --> generates the greatest shift
    
    Bob_test = Alice_test
    
    ''' The following are artificial data similar to that from [Monnier & Shevell (2004)].
    To have the accurate and complete data, please ask them directly. '''
    
    Alice_comp = np.array([[0.95, 0.98, 0.98],
                         [1. , 0.98, 0.98],
                         [1.2, 0.98, 0.98],
                         [0.7, 0.98, 0.98],
                         [0.8, 0.98, 0.98],
                         [1.1, 0.98, 0.98],
                         [1.36,0.98,0.98],
                         [0.63,0.98,0.98]])
    
    Bob_comp = np.array([[0.9, 0.98, 0.98],
                       [1.1, 0.98, 0.98],
                       [1.1, 0.98, 0.98],
                       [0.9, 0.98, 0.98],
                       [0.8, 0.98, 0.98],
                       [1.1, 0.98, 0.98],
                       [1.3, 0.98, 0.98],
                       [0.6, 0.98, 0.98]])
    
    everyone_test = np.concatenate((Alice_test,Bob_test))
    everyone_comp = np.concatenate((Alice_comp,Bob_comp))
    
    '''DETERMINES DATA (comment/uncomment where necessary) '''
    test_configs = Alice_test - col_diff
    comp_configs = Alice_comp - col_diff
    print('SIMULATING Alice')
    
#    test_configs = Bob_test- col_diff
#    comp_configs = Bob_comp - col_diff
#    print('SIMULATING Bob')
    
    print('this is an artificial dataset')
    
    C = create_images(test_configs[6])  # to check whether spatial resolution is good
    imC = plt.imshow(C,extent = [-1,1,-1,1], cmap='RdBu',vmin = -1, vmax = 1)
    plt.colorbar(imC, orientation='vertical')
    if __name__ == '__main__' :
        plt.show()


if date == 2008 :
    col_diff = 1 # to fix the position of the neutral color (new zero)
    purple, lime, test, white = np.array([1.5,0.5,1,1])-col_diff # some colors...
    loc = spatial_index(R - (nb+0.5)*dr) # point of interest when doing color matching


    Charlie_test = np.array([[0.16,1.5,0.5],
                      [0.25,1.5,0.5],
                      [0.35,1.5,0.5],
                      [1.,1.5,0.5],
                      [1.7,1.5,0.5],
                      [1.9,1.5,0.5],
                      [2.1,1.5,0.5],
                      
                      [0.16,0.5,1.5],
                      [0.25,0.5,1.5],
                      [0.35,0.5,1.5],
                      [1.,0.5,1.5],
                      [1.7,0.5,1.5],
                      [1.9,0.5,1.5],
                      [2.1,0.5,1.5]])
    
    ''' The following are artificial data similar to that from [Monnier (2008)].
    To have the accurate and complete data, please ask the author directly. '''
    
    Charlie_pl_shift = np.array([0.3,0.34,0.4,0.5,0.4,0.35,0.3])
    Charlie_lp_shift = - np.array([0.15,0.2,0.25,0.4,0.4,0.35,0.3])
    Charlie_comp = Charlie_test.copy()
    Charlie_comp[:7,0] += Charlie_pl_shift ; Charlie_comp[7:,0] += Charlie_lp_shift
    
    verif_pl = np.array([[x, 1.5,0.5] for x in np.linspace(0,2.1,15)])
    verif_lp = np.array([[x, 0.5,1.5] for x in np.linspace(0,2.1,15)])
    verif = np.concatenate((verif_pl,verif_lp)) # provides a `continuous' range of tested s values

    '''DETERMINES DATA'''
    test_configs = Charlie_test - col_diff
    comp_configs = Charlie_comp - col_diff
    verif_configs = verif - col_diff # fictive tests
    print('SIMULATING Charlie')
    print('this is an artificial dataset')
    


    C = create_images(test_configs[6])  # to check whether spatial resolution is good
    imC = plt.imshow(C,cmap='RdBu',vmin = -max(C.min(),C.max()), vmax = max(C.min(),C.max()))
    plt.colorbar(imC, orientation='vertical')
    if __name__ == '__main__' :
        plt.show()
    
if date == 2017 :
    loc = spatial_index(-R/2) # point of interest when doing color matching

    def hsl2hslxyz(h,s,l) :
        x = s/100*np.cos(2*np.pi*h/360) ; y = s/100*np.sin(2*np.pi*h/360) ; z = l/100
        return x,y,z
    
    # some colors...
    h = 60 ; s = 60 ; l = 50
    x,y,z = hsl2hslxyz(h,s,l)
    yellow = np.array([x,y,2*z-1])
    h = 0 ; s = 60 ; l = 50
    x,y,z = hsl2hslxyz(h,s,l)
    red = np.array([x,y,2*z-1])
    h = -60 ; s = 60 ; l = 50
    x,y,z = hsl2hslxyz(h,s,l)
    violet = np.array([x,y,2*z-1])
    h = 0 ; s = 0 ; l = 0
    x,y,z = hsl2hslxyz(h,s,l)
    gray = np.array([x,y,2*z-1])
    
    
    from HSL_data import cyl5 as starts, dyl5 as ends
    starts[:,2] = 2*starts[:,2] - 1 # where arrows start
    ends[:,2] = 2*ends[:,2] - 1 # where arrows end

    HSL_test_configs = np.array([[c,yellow] for c in starts])
    HSL_comp_configs = np.array([[c,gray] for c in ends])