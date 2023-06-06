import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dataloaders import generate_data_wlan_channel
from easydict import EasyDict as edict
from ga import ga
from ofa import ofa
from offline_algos import optimal_config


def compute():
    with np.printoptions(precision=3,suppress=True):
        ops_ogd={}
        ops_greedy={}
        # alphas=np.arange(0.0,1.0,0.1)
        # alphas=np.concatenate((np.append(np.arange(0,1,0.1),0.999),np.arange(2.0,20.,1)))
        # alphas=np.arange(0.,100.,10)
        # alphas=[np.array([0.5,0.5,0.99,0.0,0.2])]
        alphas=np.concatenate((np.append(np.arange(0,1,0.1),0.999),np.arange(1.1,2.1,0.1)))
        for alpha in alphas:
            y_opt,_=optimal_config(T,num_users,requests,alpha,max_iter=2000)
            print(f'{alpha:.3f}')
            ops_ogd[f'{alpha:.3f}']=ofa(T,num_users,requests,alpha,y_opt)
            ops_greedy[f'{alpha:.3f}']=ga(T,num_users,requests,alpha,y_opt)
    
    to_pickle=edict({
        'T':T,
        'num_users':num_users,
        'requst_sequence':requests,
        'ogd_output': ops_ogd,
        'greedy_output':ops_greedy,
        'description': "output_dic is the dictionary containing output of ogd algorithm for various alphas"})
    if dataset=='wlan':
        to_pickle['seed']=seed
    with open(f'{path}/{dataset}_T{T}_m{num_users}.p','wb') as f:
        pickle.dump(to_pickle,f)



if __name__=="__main__":
    dataset='mcs' #wlan
    # dataset='wlan'
    parent_folder = Path(__file__).parent
    path=f'{parent_folder}/output/{dataset}'

    if dataset=='wlan':
        T=2000
        num_users=5
        seed=10
        rng=np.random.default_rng(seed)
        requests=generate_data_wlan_channel(num_users,T,rng)
    elif dataset=='mcs':
        T=2000
        num_users=2
        folder_path=f'{parent_folder}/data'
        x1=np.loadtxt(f'{folder_path}/rate_UE200m.csv',delimiter=',')
        x2=np.loadtxt(f'{folder_path}/rate_UE400m.csv',delimiter=',') # len= is 9678
        requests=np.zeros((2,x1.shape[0]))
        requests[0,:]=x1
        requests[1,:]=x2
        requests/=requests.max()
    
    compute()