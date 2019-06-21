#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:52:17 2018

Last modified: Fri Dec 14 2018

@author: Anna SONG

Please see the related article for a global description of our algorithms.

For further information on pytorch and its usage, please refer to:
    [A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito,
    Z. Lin, A. Desmaison, L. Antiga, and A. Lerer, Automatic differentiation in pytorch,
    in NIPS-W, 2017]

main_HSL_pytorch_2D.py depends on the following scripts:
    - settings.py
    - visualize.py
    - pytorch_argmax.py
    - initialization.py
    
TO LAUNCH THE REGRESSION put descend_gradient = True

This script is essentially very close in structure to main_pytorch.py
so refer to it for further explanations in the header.

NOTES for this script:
    Here it is a 2D chromatic disk inside HSL space (and NOT 3D HSL).

"""


import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch import Tensor

from scipy.optimize import minimize
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt

from settings import R,LC,Nc,Nx,dx,dc,loc, date
if date != 2017 :
    raise ValueError('This script works only for the HSL setting. Please change variable date in settings.py to 2017.')
from settings import HSL_test_configs, HSL_comp_configs
from settings import gray, yellow, red, violet
from settings import color_index, space_abs, color_abs
from settings import create_images

from visualize import write, show_maps
from pytorch_argmax import HSL_SoftArgmax_2D, SoftMax

import time


# truncate all values to obtain 2D shifts in the chromatic plane (forget about Luminance)
gray = gray[:2]
yellow = yellow[:2]
red = red[:2]
violet = violet[:2]
HSL_test_configs = HSL_test_configs[:,:,:2]
HSL_test_configs = HSL_test_configs[::2] # take every out of 2
HSL_comp_configs = HSL_comp_configs[:,:,:2]
HSL_comp_configs = (HSL_comp_configs[::2] + HSL_comp_configs[1::2])/2 # mean value along Luminance



dtype = torch.FloatTensor


#%%
''' IMPORTANT CONFIGURATIONS '''
descend_gradient = 1 # choose False or True to launch gradient descent or not

show = False # by default, do not show the intermediate activities

# the parameters w.r.t. which we differentiate and allow gradient descent
we_diff = 'fg' # 'fg' by default (good in practice), only differentiating on parameters of f and g
if we_diff == 'fg': 
    print('We will differentiate on parameters of f and g only, but you can change this.')

if we_diff == 'fg' : # tune in physical and color space
    do_we_grad = {'muf':True, 'nuf':True, 'alphaf':True, 'betaf':True,
                  'mug':True, 'nug':True, 'alphag':True, 'betag':True,
                  'muh':False,'sigmah':False,'gamma':False}
    
elif we_diff == 'g' : # tune in physical space
    do_we_grad = {'muf':False, 'nuf':False, 'alphaf':False, 'betaf':False,
                  'mug':True, 'nug':True, 'alphag':True, 'betag':True,
                  'muh':False,'sigmah':False,'gamma':False}

elif we_diff == 'f' : # tune in color space
    do_we_grad = {'muf':True, 'nuf':True, 'alphaf':True, 'betaf':True,
                  'mug':False, 'nug':False, 'alphag':False, 'betag':False,
                  'muh':False,'sigmah':False,'gamma':False}

elif we_diff == 'hF' : # tune only input parameters and excitability
    do_we_grad = {'muf':False, 'nuf':False, 'alphaf':False, 'betaf':False,
                  'mug':False, 'nug':False, 'alphag':False, 'betag':False,
                  'muh':True,'sigmah':True,'gamma':True}

elif we_diff == 'all' : # tune all 11 parameters
    do_we_grad = {'muf':True, 'nuf':True, 'alphaf':True, 'betaf':True,
                  'mug':True, 'nug':True, 'alphag':True, 'betag':True,
                  'muh':True,'sigmah':True,'gamma':True}

else:
    raise ValueError('Please use a valid argument for we_diff')

parameter_names = ['muf','nuf','alphaf','betaf','mug','nug','alphag','betag','muh','sigmah','gamma']
subset_indices = [do_we_grad[x]*i for i,x in enumerate(parameter_names) if do_we_grad[x]]

Nexp = len(HSL_test_configs)
Ncol = (2*Nc+1)**2


def cartesian_product(*arrays):
    la = len(arrays)
    ddtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=ddtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

color_abs_prod = cartesian_product(color_abs,color_abs)


def create_diffs(C) :
    '''for a stack of input cortical images C of shape NN * (2*Nx+1) * (2*Nx+1)
    create_diffs() creates a bunch of images of the form  c - C(r)
    where c is a variable in color_abs.
    This term will be given to the gaussian h so that we obtain
    the LGN input H.
    C is here supposed to be a scalar image, not a 3D color image (and not 2D either, because we produce two independent images as below).'''
    Cdiff = color_abs[None,None,None,:] - C[:,:,:,None]
    return Variable(torch.from_numpy(Cdiff).type(dtype)) # shape NN * (2*Nx+1) * (2*Nx+1) * (2*Nc+1)


#%%
''' The main operators involved in the layers of the big operator
a \mapsto F(w * a + H) '''

def Gauss(abscisse,sigma,mu = 1):
    gauss = lambda x : mu*np.exp(-(x**2)/(2*sigma**2))
    G = gauss(abscisse)
    return G

ext_X,ext_Y = np.mgrid[-R:R:(2*Nx+1)*1j,-R:R:(2*Nx+1)*1j]

class Gaussian_Conv(Function) :
    '''2D gaussian convolution on physical space (prepares for function g)'''

    @staticmethod
    def forward(ctx,a,sigma) :
        ctx.save_for_backward(a,sigma)
        a_n, s_n = a.cpu().detach().numpy(), sigma.cpu().detach().numpy()[0]
        G = Gauss(sigma = s_n, abscisse = space_abs)
        fu = fftconvolve(a_n,G.reshape((1,2*Nx+1,1,1,1)),'same')
        fu = fftconvolve(fu,G.reshape((1,1,2*Nx+1,1,1)),'same')
        return torch.from_numpy( fu ).type(dtype)
    
    @staticmethod
    def backward(ctx,grad_output):
        a,sigma = ctx.saved_tensors

        a_n =     a.data.cpu().numpy()
        s_n = sigma.data.cpu().numpy()[0]
        grad_output_n = grad_output.data.cpu().numpy()

        G = Gauss(sigma = s_n, abscisse = space_abs)
        fu = fftconvolve(grad_output_n,G.reshape((1,2*Nx+1,1,1,1)),'same')
        fu = fftconvolve(fu,G.reshape((1,1,2*Nx+1,1,1)),'same')
        grad_a = Variable(torch.from_numpy( fu ).type(dtype))

        xy_filter = lambda x,y : (x**2+y**2)/(s_n**3)*np.exp(-(x**2+y**2)/(2*s_n**2))
        G = xy_filter(ext_X,ext_Y).reshape((1,2*Nx+1,2*Nx+1,1,1))
        aa = fftconvolve(a_n,G,'same')
        aa = Variable(torch.from_numpy(aa).type(dtype))
        grad_sigma = Tensor([torch.dot(aa.view(-1),grad_output.view(-1))])

        return grad_a,grad_sigma


def Flip_or_not(fw,flip) :
    ''' flips along the c coordinate in color space'''
    if not flip :
        return fw
    # ascontiguousarray seems necessary for a good behavior in memory
    fp = np.ascontiguousarray(np.flip(fw,3))
    fp = np.ascontiguousarray(np.flip(fp,4))
    return fp

cxx,cyy = np.mgrid[-LC:LC:(2*Nc+1)*1j,-LC:LC:(2*Nc+1)*1j]

class Gaussian_Conv_Color(Function) :
    """convolution along color axis, with flip option"""
    @staticmethod
    def forward(ctx, a,sigma,flip) :
        ctx.save_for_backward(a,sigma)
        ctx.flip = flip
                             
        a_n, s_n = a.cpu().detach().numpy(), sigma.cpu().detach().numpy()[0]
        C = Gauss(sigma = s_n, abscisse = color_abs)
        fw = fftconvolve(a_n,C.reshape((1,1,1,1,2*Nc+1)),'same')
        fw = fftconvolve(fw,C.reshape((1,1,1,2*Nc+1,1)),'same')
        fw = Flip_or_not(fw,flip)
        return torch.from_numpy(fw).type(dtype)
    
    @staticmethod
    def backward(ctx,grad_output):
        a,sigma = ctx.saved_tensors
        flip = ctx.flip
        grad_flip = None

        a_n = a.data.cpu().numpy()
        s_n = sigma.data.cpu().numpy()[0]
        grad_output_n = grad_output.data.cpu().numpy()

        C = Gauss(sigma = s_n, abscisse = color_abs)
        fw = fftconvolve(grad_output_n,C.reshape((1,1,1,1,2*Nc+1)),'same')
        fw = fftconvolve(fw,C.reshape((1,1,1,2*Nc+1,1)),'same')
        
        fw = Flip_or_not(fw,flip)
        grad_a = Variable(torch.from_numpy(fw).type(dtype))
        
        c_filter = lambda c1,c2 : ((c1**2+c2**2)/s_n**3)*np.exp(-(c1**2+c2**2)/(2*s_n**2))
        C = c_filter(cxx,cyy).reshape((1,1,1,2*Nc+1,2*Nc+1))
        fw = fftconvolve(a_n,C,'same')
        fw = Flip_or_not(fw,flip)
        fw = Variable(torch.from_numpy(fw).type(dtype))
        grad_sigma = Tensor([torch.dot(fw.view(-1),grad_output.view(-1))])
        
        return grad_a,grad_sigma,grad_flip


def Mexican_Hat(a,mug,nug,alphag,betag) :
    '''corresponding to function g'''
    GC = Gaussian_Conv().apply
    return mug * GC(a, alphag) - nug * GC(a, betag)


def SOG(a,muf,nuf,alphaf,betaf) :
    '''corresponding to function f (a non-symmetric `sum' of gaussians)'''
    GCC = Gaussian_Conv_Color().apply
    return muf * GCC(a,alphaf,False) - nuf * GCC(a,betaf,True)


class Sigmoid(nn.Module):
    '''corresponding to function F, a sigmoid function passing through 1/2 at x = 0 '''
    def __init__(self):
        super(Sigmoid,self).__init__()
        self.gamma = nn.Parameter(torch.Tensor([5.]).type(dtype),requires_grad = do_we_grad['gamma'])
    
    def forward(self,a):
        F = lambda x : 1./(1+torch.exp(-self.gamma*x))
        return F(a)

class ColorPeak(nn.Module):
    ''' corresponds to function H, the LGN input or `forcing term' '''
    def __init__(self):
        super(ColorPeak,self).__init__()
        self.muh = nn.Parameter(torch.Tensor([.2]).type(dtype),requires_grad = do_we_grad['muh'])
        self.sigmah = nn.Parameter(torch.Tensor([.5]).type(dtype),requires_grad = do_we_grad['sigmah'])
        
    def forward(self,Diff) :
        hauss = lambda x : torch.exp(-(x**2)/(2*self.sigmah**2))
        H = self.muh*hauss(Diff[0])[:,:,:,:,None]*hauss(Diff[1])[:,:,:,None,:]
        return H

#%%
        
''' This is the core of the script '''

class Net(torch.nn.Module):
    def __init__(self):
        """
        This is the main Wilson Cowan algorithm, which assembles multiple
        operations on an arbritrary initial activity a(r,c). The final activity
        a(r_0,c), where r_0 is a point of interest, comes from a(r,c) stationary
        point of a = F(a,q) (found after 15-20 iterations)
        where q is the set of parameters to be minimized.
       """
        super(Net, self).__init__()
        
        self.f = SOG
        self.muf = nn.Parameter(torch.Tensor([1.0]).type(dtype),requires_grad = do_we_grad['muf'])
        self.nuf = nn.Parameter(torch.Tensor([1.3]).type(dtype),requires_grad = do_we_grad['nuf'])
        self.alphaf = nn.Parameter(torch.Tensor([0.8]).type(dtype),requires_grad = do_we_grad['alphaf'])
        self.betaf = nn.Parameter(torch.Tensor([0.9]).type(dtype),requires_grad = do_we_grad['betaf'])
        
        self.g = Mexican_Hat
        self.mug = nn.Parameter(torch.Tensor([3.0]).type(dtype),requires_grad = do_we_grad['mug'])
        self.nug = nn.Parameter(torch.Tensor([3.0]).type(dtype),requires_grad = do_we_grad['nug'])
        self.alphag = nn.Parameter(torch.Tensor([0.2]).type(dtype),requires_grad = do_we_grad['alphag'])
        self.betag = nn.Parameter(torch.Tensor([0.5]).type(dtype),requires_grad = do_we_grad['betag'])

        self.F = Sigmoid()
        self.H = ColorPeak()
        
        self.Argmax = HSL_SoftArgmax_2D()    
        self.Max = SoftMax()

    def forward(self,Diff,loc,N = Nexp):
        a = Variable(.5*torch.ones(N+Ncol,2*Nx+1,2*Nx+1,2*Nc+1,2*Nc+1))
        print('a.shape',a.shape)
        for _ in range(15) : # 15 to 20 iterations are sufficient for the activities to converge
            a = self.g(a,self.mug,self.nug,self.alphag,self.betag)
            a = self.f(a,self.muf,self.nuf,self.alphaf,self.betaf)
            a = dc**2*dx**2*a # do not forget for a real integration
            a = self.F(a + self.H(Diff))
        a = a[:,Nx,loc,:,:] # we restrict to the point of interest r_0, indicated by loc from settings.py
        a_tests = a[:N]  # shape N * (2*Nc+1) * (2*Nc+1); for all the test patterns of the experiments
        as_compared = a[N:] # shape Ncol * (2*Nc+1) * (2*Nc+1); all the comparison patterns have to be tested (same family is used for all experiments)
        # here Ncol = (2*Nc+1)**2 2D color space
        u = torch.abs(as_compared[None,:,:,:] - a_tests[:,None,:,:]) # shape (N = Nexp,Ncol,(2*Nc+1),(2*Nc+1))
        u = u.view(N,Ncol,(2*Nc+1)**2)
        u = self.Max(u) # shape (N = Nexp,Ncol)
        u = 1/(u+1) # important to look for min instead of max
        comp_preds = self.Argmax(u,color_abs_prod) # shape (Nexp,2)
        return comp_preds, a_tests, as_compared

    def from_numpy(self, q):
        ''' This feeds the model.parameters with a new value for q
        when it is a numpy array '''
        muf,nuf,alphaf,betaf,mug,nug,alphag,betag,muh,sigmah,gamma = q
        self.muf.data = torch.from_numpy(np.array([muf])).type(dtype)
        self.nuf.data = torch.from_numpy(np.array([nuf])).type(dtype)
        self.alphaf.data = torch.from_numpy(np.array([alphaf])).type(dtype)
        self.betaf.data = torch.from_numpy(np.array([betaf])).type(dtype)
        self.mug.data = torch.from_numpy(np.array([mug])).type(dtype)
        self.nug.data = torch.from_numpy(np.array([nug])).type(dtype)
        self.alphag.data = torch.from_numpy(np.array([alphag])).type(dtype)
        self.betag.data = torch.from_numpy(np.array([betag])).type(dtype)
        self.H.muh.data = torch.from_numpy(np.array([muh])).type(dtype)
        self.H.sigmah.data = torch.from_numpy(np.array([sigmah])).type(dtype)
        self.F.gamma.data = torch.from_numpy(np.array([gamma])).type(dtype)

HSLmodel = Net()

#%%

def negative_loss(p) :
    return torch.min(Variable(torch.zeros(1),requires_grad = False),p)**2

def subset_parameters(HSLmodel) :
    return [p for p in HSLmodel.parameters() if p.requires_grad]

criterion = lambda c1,c2 : ((c1 - c2)**2).sum()
tnf = 0

def Loss_and_grads(HSLmodel,q) :
    ''' Loss_and_grads(q) returns two values: the current loss, and the gradients
    corresponding to parameters wrt which we want to differentiate.
    It relies on the model = Net() created using pytorch.
    This wraps the input used to feed scipy.optimize.minimize
    '''
    
    print('\n')
    write(q)

    global tnf
    tnf += 1
    
    loss = 0
    
    HSLmodel.from_numpy(q)
    all_configs = np.concatenate((HSL_test_configs,np.array([[compared,gray] for compared in color_abs_prod])))
    C1 = create_images(all_configs[:,:,0])
    Cdiff1 = create_diffs(C1)
    C2 = create_images(all_configs[:,:,1])
    Cdiff2 = create_diffs(C2)
    Ddiff = torch.cat((Cdiff1[None],Cdiff2[None])) # 2 stacks of color images for the 2D color space<

    comp_preds,a_tests,_ = HSLmodel(Ddiff,loc)
    comps = HSL_comp_configs[:,0,:] # shape (Nexp,2)

    # what we want to optimize
    loss += 100*criterion(comp_preds,Variable(torch.from_numpy(comps).type(dtype)))
    print('tnf = %.0f  current loss: ' %tnf, loss.item())
    
    if tnf%3 == 1:
        pred_results = comp_preds.data.numpy()
        pred_X = pred_results[:,0]
        pred_Y = pred_results[:,1]
        X = HSL_test_configs[:,0,0] ; Y = HSL_test_configs[:,0,1]
        U = pred_X - X ; V = pred_Y - Y
        plt.figure()
        plt.axis('equal')    
        plt.quiver(X,Y,U,V,units = 'xy',scale = 1)
        plt.show()
    

    # now it's time to differentiate
    parameters = subset_parameters(HSLmodel)
    dq_loss = torch.autograd.grad(loss,parameters)
    dq_loss_numpy = np.zeros_like(q)
    for i,x in enumerate(subset_indices) :
        dq_loss_numpy[x] = dq_loss[i]
    print('dq_loss_numpy',np.round(dq_loss_numpy,3)) # to check the gradient
    return loss.data.numpy().astype('float64'), dq_loss_numpy.astype('float64')



def grad_desc(q0,show = False) :
    ''' Wrap the result of the minimization and show the graphs to understand
    the computed q_localmin '''
    res = minimize(lambda x : Loss_and_grads(HSLmodel,x), # function to minimize
						q0, # starting estimate
						method = 'L-BFGS-B',  # an order 2 method
						jac = True,           # matching_problems also returns the gradient
						options = dict(maxiter = 60,ftol = 1e-3, gtol = 1e-3, maxcor = 10, disp = True))
#    print('final gradients')
#    for y in res.jac :
#        print(y)
    print('final values')
    for y,name in zip(res.x,parameter_names) :
        print(name,y)
    #if show :
        #show_maps(res.x)
    write(res.x,letter = False)
    return res


#%%
   
def show_HSL_comp(q) :
    '''Displays simulated shifts in the chromatic disk. '''
    HSLmodel.from_numpy(q)
    
    all_configs = np.concatenate((HSL_test_configs,np.array([[compared,gray] for compared in color_abs_prod])))
    C1 = create_images(all_configs[:,:,0])
    Cdiff1 = create_diffs(C1)
    C2 = create_images(all_configs[:,:,1])
    Cdiff2 = create_diffs(C2)
    Ddiff = torch.cat((Cdiff1[None],Cdiff2[None])) # for 2D color images diffs
    
    comp_preds,a_tests,as_compared = HSLmodel(Ddiff,loc)
    pred_results = comp_preds.data.numpy()

    pred_X = pred_results[:,0]
    pred_Y = pred_results[:,1]
    X = HSL_test_configs[:,0,0] ; Y = HSL_test_configs[:,0,1]
    U = pred_X - X ; V = pred_Y - Y

     
    plt.figure()
    plt.quiver(X,Y,U,V,units = 'xy',scale = 1)
        
    plt.scatter(yellow[0],yellow[1],s = 15, c ='y')
    plt.scatter(red[0],red[1],s = 15, c ='r')
    plt.scatter(violet[0],violet[1],s = 15, c ='k')

    plt.axis('equal')
    plt.title('predicted data with \n q = ' + 'f %.2f, %.2f, %.2f, %.2f, g %.2f, %.2f, %.2f, %.2f, h %.2f, %.2f, F %.2f' % tuple(q) )
    plt.show()
    
    a_test = a_tests[0].data.numpy()
    comp = comp_preds[0].data.numpy()
    i,j = int(color_index(comp[0])),int(color_index(comp[1]))
    k = np.where( (i == color_index(color_abs_prod[:,0])) *  (j == color_index(color_abs_prod[:,1])))
    a_comp = as_compared[k].data.numpy()[0]
#        
#    plt.figure()
#    ima_test = plt.imshow(a_test,cmap = 'RdBu',vmin = 0,vmax = 1)
#    plt.colorbar(ima_test, orientation='vertical')
#    plt.show()
#    
#    plt.figure()
#    ima_comp = plt.imshow(a_comp,cmap = 'RdBu',vmin = 0,vmax = 1)
#    plt.colorbar(ima_comp, orientation='vertical')
#    plt.show()
    
    return pred_results,U,V,a_test,a_comp


def show_HSL_exp() :
    '''Displays real experimental shifts in the chromatic disk. '''

    X = HSL_test_configs[:,0,0] ; Y = HSL_test_configs[:,0,1]
    comp_X = HSL_comp_configs[:,0,0] ; comp_Y = HSL_comp_configs[:,0,1]
    U = comp_X - X ; V = comp_Y - Y
    
    plt.figure()
    plt.quiver(X,Y,U,V,units = 'xy',scale = 1)
    plt.scatter(yellow[0],yellow[1],s = 15, c ='y')
    plt.scatter(red[0],red[1],s = 15, c ='r')
#    plt.scatter(violet[0],violet[1],s = 15, c ='k')

    plt.axis('equal')
    plt.title('experimental data')
    plt.show()


if __name__ == '__main__' :
    # for the default values, Nx = 4, Nc = 4, do_we_grad = 'fg',
    # GD will typically take about 10 minutes so it is quite long: be patient!
    # this is in part due to the large dimension of the problem:
    # note that the tensor of all activities a(...) is of dimension (117,9,9,9,9)
    # and contains  elements! That is why increasing the sampling resolution
    # may lead to ever lasting computations.

    from initialization import q_HSL as q # this value already close to optimal, you can change it
    show_maps(q)
    show_HSL_exp()
    print('lauching the regression')
    pred_results,U,V,a_test,a_comp = show_HSL_comp(q)

    if descend_gradient : # wait for ~10 mn (depending on your computer)
        t = time.clock()
        res = grad_desc(q,show = True)
        print(time.clock() - t, "seconds to make GD")
        show_HSL_exp()
        pred_results,U,V,a_test,a_comp = show_HSL_comp(res.x)
