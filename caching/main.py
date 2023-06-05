import pickle
from pathlib import Path

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

def compute():
    with np.printoptions(precision=3,suppress=True):
        ops={}
        ops_salem={}
        alphas=np.array([0,0.1])
        # alphas=np.concatenate((np.append(np.arange(0,1,0.1),0.999),np.arange(1.1,2.1,0.1)))
        y_maximin,_=maximin_optimal_config(N,T,num_users,cache_size,requests)
        ops_maximin=comp_hitrate(N,T,num_users,cache_size,requests,y_maximin)
        for alpha in alphas:
            y_opt,_=optimal_config_alpha_util(N,T,num_users,cache_size,requests,alpha)
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


def set_theme():
    sns.set_theme(style="white")
    sns.set_context("notebook",rc={"lines.linewidth": 2.5,'font.size': 12.0,
                               'xtick.labelsize': 15.0, 'ytick.labelsize': 15.0,
                               'axes.labelsize': 16.0, 'axes.titlesize': 16.0,
                               'legend.fontsize': 12.0})

def genearate_plot_avg_hitrate():
    set_theme()
    if dataset=='cmu':
        from_pickle=pickle.load(open(f'{path}/{dataset}_T{T}_N{N}_m{num_users}_k{cache_size}.p','rb'))
    else:
        from_pickle=pickle.load(open(f'{path}/{dataset}_T{T}_N{N}_m{num_users}_k{cache_size}_seed{seed}.p','rb'))

    # #### output_dic keys: dict_keys(['hitrates', 'hitrates_optimal', 'regret', 'c_regret', 'jains_index'])
    ops=from_pickle.ops
    ops_lru=from_pickle.ops_lru
    ops_lfu=from_pickle.ops_lfu
    # maximin_hitrate=from_pickle.maximin_hitrate
    ops_maximin=from_pickle.ops_maximin
    requests=from_pickle.requst_sequence
    # ops_diff_alphas=from_pickle.ops_diff_alphas
    ops_salem=from_pickle.ops_salem

    means_opf=[]
    means_salem=[]
    means_opf_sampled=[]
    means_opf_offline=[]
    alphas=[]
    for (k,op) in sorted(ops.items(),key= lambda x:float(x[0])):
        alpha=float(k)
        # if alpha>1:
        #     break
        alphas.append(alpha)
        s=0
        for i in range(num_users):
            s+=op.hitrates[i][-1]
        means_opf.append(s/num_users)
  
        s=0
        for i in range(num_users):
            s+=op.hitrates_sampled[i][-1]
        means_opf_sampled.append(s/num_users)

        s=0
        for i in range(num_users):
            s+=op.hitrates_optimal[i][-1]
        means_opf_offline.append(s/num_users)

        s=0
        for i in range(num_users):
            s+=ops_salem[k].hitrates[i][-1]
        means_salem.append(s/num_users)

    fig=plt.figure()
    x=100
    l1, =plt.plot(alphas[:x],means_opf[:x],linewidth=2.5,label='OFA')
    l2, =plt.plot(alphas[:x],means_salem[:x],linewidth=2.5,label='Si Salem et al.')
    # plt.plot(alphas[:x],means_opf_sampled[:x],linewidth=2.5,label='OPF Sampled')
    l3, =plt.plot(alphas[:x],means_opf_offline[:x],linewidth=2.5,label='Offline Optimal')

    s=0
    for i in range(num_users):
        s+=ops_lfu.hitrates[i][-1]
    mean_lfu=s/num_users

    s=0
    for i in range(num_users):
        s+=ops_lru.hitrates[i][-1]
    mean_lru=s/num_users

    s=0
    for i in range(num_users):
        s+=ops_maximin.hitrates[i][-1]
    mean_maximin=s/num_users

    l4=plt.hlines(y=mean_lru,color='r',ls=':',label='LRU',xmin=min(alphas),xmax=max(alphas))
    l5=plt.hlines(y=mean_lfu,color='k',ls=':',label='LFU',xmin=min(alphas),xmax=max(alphas))
    l6=plt.hlines(y=mean_maximin,color='b',ls=':',label='Maximin',xmin=min(alphas),xmax=max(alphas))
    plt.xlabel('$\\alpha$')
    plt.ylabel('Average Hitrates')
    plt.savefig(path+f'/{dataset}_avg_hitrates_vs_alpha_T{T}_N{N}_m{num_users}_k{cache_size}_seed{seed}.pdf',bbox_inches="tight")

    # generate legend separately
    lines=[l1,l2,l3,l4,l5,l6]
    legendFig = plt.figure("Legend plot")
    legendFig.legend([l1, l2, l3, l4,l5,l6], [l.get_label() for l in lines], loc='center',ncol=len(lines))
    legendFig.savefig(path+f'/{dataset}_legend_h.pdf',bbox_inches="tight")

def load_output():
    if dataset=='cmu':
        from_pickle=pickle.load(open(f'{path}/{dataset}_T{T}_N{N}_m{num_users}_k{cache_size}.p','rb'))
    else:
        from_pickle=pickle.load(open(f'{path}/{dataset}_T{T}_N{N}_m{num_users}_k{cache_size}_seed{seed}.p','rb'))

    # #### output_dic keys: dict_keys(['hitrates', 'hitrates_optimal', 'regret', 'c_regret', 'jains_index'])
    ops=from_pickle.ops
    ops_lru=from_pickle.ops_lru
    ops_lfu=from_pickle.ops_lfu
    # maximin_hitrate=from_pickle.maximin_hitrate
    ops_maximin=from_pickle.ops_maximin
    requests=from_pickle.requst_sequence
    # ops_diff_alphas=from_pickle.ops_diff_alphas
    ops_salem=from_pickle.ops_salem
    return ops,ops_lru,ops_lfu,ops_maximin,ops_salem,requests

def generate_plot_min_hitrate():
    set_theme()
    ops,ops_lru,ops_lfu,ops_maximin,ops_salem,requests = load_output()
    plt.figure()
    x=[]
    y=[]
    z=[]
    s=[]
    m=[]
    # plt.axhline(y=np.min(minimax[:,-1]),color='r',ls=':',label='maximin optimal')
    # for (k,op) in sorted(output_dic.items(),key= lambda x:float(x[0])):
    for (k,op) in ops.items():#sorted(output_dic.items(),key= lambda x:float(x[0])):
        # print(k)
        alpha=float(k)
        # if alpha>1:
        #     break
        x+=[alpha]
        
        a=np.array(op.hitrates)
        minhitrate=np.min(a[:,-1])
        y+=[minhitrate]
        
        b=np.array(op.hitrates_optimal)
        minhitrate1=np.min(b[:,-1])
        z+=[minhitrate1]

        c=np.array(ops_salem[k].hitrates)
        minhitrate_salem=np.min(c[:,-1])
        s+=[minhitrate_salem]

        d=np.array(op.hitrates_sampled)
        minhitrate1=np.min(d[:,-1])
        m+=[minhitrate1]

    plt.plot(x,y,label='OPF')
    plt.plot(x,s,label='Si Salem et al.')
    plt.plot(x,z,label='Offline Optimal')
    # plt.plot(x,m,label='OPF Sampled')

    lru=np.array(ops_lru.hitrates)
    min_lru=np.min(lru[:,-1])
    print(lru[:,-1])

    lfu=np.array(ops_lfu.hitrates)
    min_lfu=np.min(lfu[:,-1])
    print(lfu[:,-1])

    plt.hlines(y=min_lru,color='r',ls=':',label='LRU',xmin=min(x),xmax=max(x))
    plt.hlines(y=min_lfu,color='k',ls=':',label='LFU',xmin=min(x),xmax=max(x))


    # print(alpha,minhitrate)
    plt.hlines(y=np.min(ops_maximin.hitrates[:,-1]),color='b',ls=':',label='Maximin',xmin=min(x),xmax=max(x))
    # minimax=np.array(ops_minimax.hitrates)
    # plt.legend(title='Policy')
    # plt.title(f'Fair Caching')
    plt.xlabel('$\\alpha$')
    plt.ylabel('Minimum Hitrates')
    # plt.ylim([0,0.45])
    # if dataset=='cmu':
    #     plt.xlim([0,20])
    plt.savefig(path+f'/{dataset}_maximin_hitrates_T{T}_N{N}_m{num_users}_k{cache_size}_seed{seed}.pdf',bbox_inches="tight")

def generate_plot_jains_index():
    set_theme()
    ops,ops_lru,ops_lfu,ops_maximin,ops_salem,requests = load_output()
    xs=[]
    y1s=[]
    y2s=[]
    y3s=[]
    for (k,op) in sorted(ops.items(),key= lambda x:float(x[0])):
        alpha=float(k)
        # if alpha>1:
        #     break
        xs.append(alpha)
        y1s.append(op.jains_index[-1])
        y3s.append(op.jains_index_opt[-1])
        y2s.append(ops_salem[k].jains_index[-1])
    plt.figure()
    plt.plot(xs,y1s,label='OPF')
    plt.plot(xs,y2s,label='Si Salem et al.')
    plt.plot(xs,y3s,label='Offline Optimal')
    # plt.legend()
    plt.xlabel('$\\alpha$')
    # plt.title('Fair Caching')
    plt.ylabel('Jain\'s Index')
    # plt.show()
    # plt.title('Fair Scheduling')
    # plt.ylim((0.95,1.))
    # # plt.xlim(left=-0.5)
    # plt.axhline(y=ops_lru.jains_index[-1],color='r',ls=':',label='LRU')
    # plt.axhline(y=ops_lfu.jains_index[-1],color='k',ls=':',label='LFU')
    plt.hlines(y=ops_lru.jains_index[-1],color='r',ls=':',label='LRU',xmin=min(xs),xmax=max(xs))
    plt.hlines(y=ops_lfu.jains_index[-1],color='k',ls=':',label='LFU',xmin=min(xs),xmax=max(xs))
    plt.hlines(y=ops_maximin.jains_index[-1],color='b',ls=':',label='LFU',xmin=min(xs),xmax=max(xs))
    # if dataset=='cmu':
    #     plt.xlim([0,20])
    plt.ylim([0.,1.05])
    # plt.legend(title='Policy')
    plt.savefig(path+f'/{dataset}_jains_index_vs_alpha_T{T}_m{num_users}_seed{seed}.pdf',bbox_inches="tight")


if __name__=="__main__":

    dataset='cmu'
    # dataset = 'synthetic'
    parent_folder = Path(__file__).parent
    path=f'{parent_folder}/output/{dataset}'

    N=T=num_users=cache_size=requests=None
    if dataset=='cmu':
        #### import cmu data
        N=50
        T=400
        seed=50
        num_users=4
        cache_size=10
        # folder_path='data'
        folder_path=f'{parent_folder}/data'
        requests_raw=load_cmu_data2(num_users, N,folder_path=folder_path).T
        requests=requests_raw[:,:T]

    elif dataset=='synthetic':
        N=30
        T=1000
        num_users=5
        cache_size=7
        seed=50
        #### Generate Data
        rng=np.random.default_rng(seed)
        requests=load_synthetic_data()

    #### uncomment the following functions
    compute()
    genearate_plot_avg_hitrate()
    generate_plot_min_hitrate()
    generate_plot_jains_index()