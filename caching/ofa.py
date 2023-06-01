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
from memoization import CachingAlgorithmFlag, cached
from scipy.special import entr
from tools import comp_hitrate, ogd_projection_cvxpy, ogd_projection_fm, ogd_projection_scipy, madow_sampling
from tqdm import tqdm


def ogd_alpha_util(N,T,num_users,cache_size,requests,alpha,y_opt,projection_func='scipy'):

    if projection_func == 'scipy':
        projection=ogd_projection_scipy
    elif projection_func == 'cvxpy':
        projection=ogd_projection_cvxpy
    elif projection_func == 'fw':
        projection=ogd_projection_fm

    #### check if alphas are different for each user
    if isinstance(alpha,(float,int)):
        a=np.zeros(num_users)
        a[:]=float(alpha)
        alpha=a

    R_opt=np.ones(num_users)
    R=np.ones(num_users)
    R_sampled=np.ones(num_users)
    
    hitrates_optimal=[]
    hitrates=[]
    hitrates_sampled=[]
    for i in range(num_users):
        hitrates.append(np.zeros(T))
        hitrates_optimal+=[np.zeros(T)]
        hitrates_sampled+=[np.zeros(T)]
    
    cum_sq_reward=0

    y_t=np.array([cache_size/N]*N)
    cum_surrogate_reward=np.zeros(N)


    regret=np.zeros(T)
    c_regret=np.zeros(T)
    phi_diff_sampled=np.zeros(T)
    jains_index=np.zeros(T)
    jains_index_opt=np.zeros(T)
    downloads=np.zeros(T)

    online_reward=0
    optimal_reward=0

    u_t=np.zeros(N)
    for t in tqdm(range(T)):

        ### "play" the current configuration
        ### update the fractional hits at time t
        for i in range(num_users):  
            R[i]+=y_t[requests[i][t]]
            hitrates[i][t]=(R[i]-1)/(t+1)

            R_opt[i]+=y_opt[requests[i][t]]
            hitrates_optimal[i][t]=(R_opt[i]-1)/(t+1)

            y_sampled = madow_sampling(y_t,cache_size)
            R_sampled[i]+=y_sampled[requests[i][t]]
            hitrates_sampled[i][t]=(R_sampled[i]-1)/(t+1)

        regret[t]= np.sum(np.power(R_opt,1-alpha)) - np.sum(np.power(R,1-alpha))
        c_regret[t]= np.sum(np.power(R_opt,1-alpha)) - 1.445*np.sum(np.power(R,1-alpha))
        phi_diff_sampled[t] = np.sum(np.power(R,1-alpha)) - np.sum(np.power(R_sampled,1-alpha))
        jains_index[t]=np.sum(R)**2/(num_users*np.sum(np.power(R,2)))
        jains_index_opt[t]=np.sum(R_opt)**2/(num_users*np.sum(np.power(R_opt,2)))

        ### compute the next configuration
        ### compute surrogate reward using the fractional hits
        g = np.zeros(N)
        for i in range(num_users):
            g[requests[i][t]]+=1.0/np.power(R[i],alpha[i])
        cum_surrogate_reward+=g

        ### optimal eta
        cum_sq_reward+=np.sum(np.square(g))
        # D=np.sqrt(2*num_users)
        # eta_t=np.sqrt(2)*D/(2*np.sqrt(cum_sq_reward))
        eta_t=cache_size/np.sqrt(cum_sq_reward)
        # eta_t=0.01

        ### projection step
        ### cache configuration for next iteration
        u_t=y_t+eta_t*g
        # y_t=y.value
        # y_t=ogd_projection(u_t,N,T,num_users,cache_size)
        # y_t=ogd_projection_scipy(u_t,N,T,num_users,cache_size)
        y_t_prev=y_t
        y_t = projection(u_t,N,T,num_users,cache_size)

        difference=y_t - y_t_prev
        downloads[t]=np.sum(difference[difference>0])

    return edict({'hitrates':np.array(hitrates), 'hitrates_optimal':np.array(hitrates_optimal), 
                  'regret':regret, 'c_regret':c_regret, 'jains_index':jains_index, 'jains_index_opt':jains_index_opt,
                  'phi_diff_sampled':phi_diff_sampled, 'hitrates_sampled':hitrates_sampled,
                  'downloads':downloads})