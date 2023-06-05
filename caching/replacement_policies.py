import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
from memoization import cached, CachingAlgorithmFlag

def replacement_policy(N,T,num_users,cache_size,requests,policy='lru'):

    @cached(max_size=cache_size, algorithm=CachingAlgorithmFlag.LRU)
    def c_lru(x):
        return x
    @cached(max_size=cache_size, algorithm=CachingAlgorithmFlag.LFU)
    def c_lfu(x):
        return x
    @cached(max_size=cache_size, algorithm=CachingAlgorithmFlag.FIFO)
    def c_fifo(x):
        return x

    algo=None
    if policy=='lru':
        algo=c_lru
    elif policy=='lfu':
        algo=c_lfu
    else:
        algo=c_fifo

    R=np.ones(num_users)
    jains_index=np.zeros(T)

    hitrates=[]
    for i in range(num_users):
        hitrates.append(np.zeros(T))

    y_t_prev=np.zeros(N)
    downloads=np.zeros(T)
    for t in tqdm(range(T)):
        y_t=np.zeros(N)
        y_t[list(algo.cache_results())]=1
        difference=y_t - y_t_prev
        downloads[t]=np.sum(difference[difference>0])
        for i in range(num_users):
            algo(requests[i][t])
            R[i]+=y_t[requests[i][t]]
            hitrates[i][t]=(R[i]-1)/(t+1)

        jains_index[t]=np.sum(R)**2/(num_users*np.sum(np.power(R,2)))
        y_t_prev=y_t

    return edict({'hitrates':hitrates, 'downloads':downloads,
                  'jains_index':jains_index})