import pickle
from pathlib import Path

import numpy as np
from dataloaders import load_cmu_data2
from easydict import EasyDict as edict
from ofa import ogd_alpha_util
from offline_algos import maximin_optimal_config, optimal_config_alpha_util
from replacement_policies import replacement_policy
from si_salem import si_salem_algo2
from tools import comp_hitrate


def load_synthetic_data():
    reqs=[]
    reqs.append(rng.choice(N,T)) # user 1
    reqs.append(rng.choice(N,T)) # user 2
    reqs.append(np.array(list(range(0,4))*(T//4 +1))[:T]) # user 3
    reqs.append(np.array(list(range(4,15))*(T//9 +1))[:T]) # user 4
    reqs.append(np.array(list(range(15,30))*(T//5 +1))[:T]) # user 5
    return reqs

if __name__=="__main__":

    # dataset='cmu'
    dataset = 'synthetic'
    parent_folder = Path(__file__).parent
    path=f'{parent_folder}/output'

    N=T=num_users=cache_size=requests=None
    if dataset=='cmu':
        #### import cmu data
        N=50
        T=4
        seed=50
        num_users=4
        cache_size=10
        # folder_path='data'
        folder_path=f'{parent_folder}/data'
        requests_raw=load_cmu_data2(num_users, N,folder_path=folder_path).T
        requests=requests_raw[:,:T]

    elif dataset=='synthetic':
        N=30
        T=10
        num_users=5
        cache_size=7
        seed=50
        #### Generate Data
        rng=np.random.default_rng(seed)
        requests=load_synthetic_data()

    with np.printoptions(precision=3,suppress=True):
        ops={}
        ops_salem={}
        alphas=np.array([0,0.1])
        # alphas=np.concatenate((np.append(np.arange(0,1,0.1),0.999),np.arange(1.1,2.1,0.1)))
        y_maximin,v=maximin_optimal_config(N,T,num_users,cache_size,requests)
        ops_maximin=comp_hitrate(N,T,num_users,cache_size,requests,y_maximin)
        for alpha in alphas:
            y_opt,r=optimal_config_alpha_util(N,T,num_users,cache_size,requests,alpha)
            print(f'{alpha:.3f}')
            ops[f'{alpha:.3f}']=ogd_alpha_util(N,T,num_users,cache_size,requests,alpha,y_opt)
            ops_salem[f'{alpha:.3f}']=si_salem_algo2(N,T,num_users,cache_size,requests,alpha,y_opt)

        ops_lru=replacement_policy(N,T,num_users,cache_size,requests,policy='lru')
        ops_lfu=replacement_policy(N,T,num_users,cache_size,requests,policy='lfu')

    to_pickle=edict({
    'N':N,
    'T':T,
    'num_users':num_users,
    'cache_size':cache_size,
    'requst_sequence':requests,
    'ops_maximin': ops_maximin,
    'ops': ops,
    # 'ops_diff_alphas':ops_diff_alphas,
    'ops_lru':ops_lru,
    'ops_lfu':ops_lfu,
    'ops_salem':ops_salem,
    'description': "output_dic is the dictionary containing output of ogd algorithm for various alphas"})

    if dataset=='synthetic':
        to_pickle['seed']=seed

    filename=None
    if dataset=='cmu':
        filename=f'{path}/{dataset}_T{T}_N{N}_m{num_users}_k{cache_size}.p'
    else:
        filename=f'{path}/{dataset}_T{T}_N{N}_m{num_users}_k{cache_size}_seed{seed}.p'

    with open(filename,'wb') as f:
        pickle.dump(to_pickle,f)
