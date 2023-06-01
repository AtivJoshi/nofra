import copy
import os.path
import pickle

import cvxpy as cp
import h5py
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as sopt
import seaborn as sns
from easydict import EasyDict as edict
from scipy.special import entr
from tqdm import tqdm
from memoization import cached, CachingAlgorithmFlag

def optimal_config_alpha_util(N,T,num_users,cache_size,requests,alpha,max_iter=2000,method='fw'):

    def optimal_config_cvxpy(N,T,num_users,cache_size,requests,alpha):
        '''
        uses cvxpy
        N: library size
        T: horizon
        num_users: number of users
        cache_size: cache size
        requests: list of request sequence of each user
        alpha: utility parameter (float or list/array of floats)
        The function computes the optimal cache configuration according to the
        alpha utility function for the fair caching, i.e.:
        $$\max_{y^*}\sum_i \phi(⟨\sum_{\tau=1}^T x_i(\tau),y^*⟩)$$
        '''

        if isinstance(alpha,float):
            a=np.zeros(num_users)
            a[:]=alpha
            alpha=a

        y=cp.Variable(N,name='y') # variables corresponding to cache configuration
        R=[1]*num_users
        reward=0
        constr=[] # constraints
        
        #compute cumulative file requests vectors for each user
        X_T = []
        for i in range(num_users):
            arr=np.zeros(N,dtype=int)
            f,c=np.unique(requests[i],return_counts=True)
            # print('f: ',f,'c: ', c)
            arr[f]=c
            # print(arr,'\n\n')
            X_T.append(arr)
        
        for i in range(num_users):
            reward+=cp.power(X_T[i]@y,1-alpha[i])/(1-alpha[i])
        for i in range(N):
            constr+=[y[i]>=0,y[i]<=1]
        constr+=[cp.sum(y)==cache_size]
        problem=cp.Problem(cp.Maximize(reward),constr)
        problem.solve(solver='SCS')

        return y.value,np.array(X_T)@y.value

    def optimal_config_fw(N,T,num_users,cache_size,requests,alpha,max_iter=2000):
        '''
            Use Frank-Wolfe to compute the offline optimal configuration
        '''
        if isinstance(alpha,float):
            a=np.zeros(num_users)
            a[:]=alpha
            alpha=a

        #### compute cumulative file requests vectors for each user
        X_T = []
        for i in range(num_users):
            arr=np.zeros(N,dtype=int)
            f,c=np.unique(requests[i],return_counts=True)
            # print('f: ',f,'c: ', c)
            arr[f]=c
            # print(arr,'\n\n')
            X_T.append(arr)
            
        #### initialize y_k
        y_k=np.array([num_users/N]*N)
        
        for step in range(max_iter):
            # print(y_k)
            #### computing the gradient at y_k
            df_y_k = np.zeros(N)
            for i in range(num_users):
                # df_y_k+=((1-alpha)/(np.dot(X_T[i],y_k)**(alpha)))*X_T[i]
                df_y_k+=((-1.)/(np.dot(X_T[i],y_k)**(alpha[i])))*X_T[i]

            
            # print(df_y_k)
            #### compute z_k by sorting f_y_k
            idx=df_y_k.argsort()
            # print(idx)
            z_k=np.zeros(N)
            z_k[idx[:cache_size]]=1
            # print(z_k)
            eta=(2./(step+3))
            y_prev=y_k
            y_k=y_k+eta*(z_k-y_k)
            # print(np.max(np.absolute(y_prev-y_k)))
        
        return y_k,np.array(X_T)@y_k
    
    if method=='fw':
        return optimal_config_fw(N,T,num_users,cache_size,requests,alpha,max_iter)
    elif method=='cvxpy':
        return optimal_config_cvxpy(N,T,num_users,cache_size,requests,alpha)
    else:
        raise ValueError('invalid argument')


def maximin_optimal_config(N,T,num_users,cache_size,requests):

    X_T = []
    for i in range(num_users):
        arr=np.zeros(N,dtype=int)
        f,c=np.unique(requests[i],return_counts=True)
        # print('f: ',f,'c: ', c)
        arr[f]=c
        # print(arr,'\n\n')
        X_T.append(arr)
    # print(X_T)
    X=np.array(X_T)
    # print(X)

    constr=[]
    y=cp.Variable(N,name='y')
    v=cp.Variable(1,name='v')

    for i in range(N):
        constr+=[y[i]>=0,y[i]<=1]
    constr+=[cp.sum(y)==cache_size]

    for i in range(num_users):
        constr+=[X_T[i]@y>=v]
    
    problem=cp.Problem(cp.Maximize(v),constr)
    problem.solve()

    return y.value, v.value
# with np.printoptions(precision=3,suppress=True):
#     print(minimax_optimal_config(N,T,num_users,cache_size,requests))