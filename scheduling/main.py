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

def set_theme():
    sns.set_theme(style="white")
    sns.set_context("notebook",rc={"lines.linewidth": 2.5,'font.size': 12.0,
                               'xtick.labelsize': 15.0, 'ytick.labelsize': 15.0,
                               'axes.labelsize': 16.0, 'axes.titlesize': 16.0,
                               'legend.fontsize': 12.0})


def load_output():
    from_pickle=pickle.load(open(f'{path}/{dataset}_T{T}_m{num_users}.p','rb'))
    ops_ofa=from_pickle.ogd_output
    ops_ga=from_pickle.greedy_output
    requests=from_pickle.requst_sequence
    return ops_ofa, ops_ga, requests

def genearate_plot_jains_index():
    set_theme()
    ops_ofa, ops_ga, requests=load_output()
    xs=[]
    y1s=[]
    y2s=[]
    for (k,op) in sorted(ops_ofa.items(),key= lambda x:float(x[0])):
        alpha=float(k)
        # if alpha>1:
        #     break
        xs.append(alpha)
        y1s.append(op.jains_index[-1],)
        y2s.append(ops_ga[k].jains_index[-1])
    plt.figure(figsize=(6.67,5))
    plt.plot(xs,y1s,label='OFA')
    plt.plot(xs,y2s,label='GA')
    plt.legend()
    plt.xlabel('$\\alpha$')
    plt.ylabel('Jain\'s Index')
    # plt.title('Fair Scheduling')
    # plt.xlim([0,2])
    # plt.xticks(np.arange(0,2.01,0.5))
    plt.ylim((0,1.01))
    # with np.printoptions(precision=3,suppress=True):
    #     print(np.array(xs)[:10])
    #     print(f'OFA:\t',np.array(y1s)[:10])
    #     print(f'GA:\t',np.array(y2s)[:10])    
    plt.savefig(path+f'/{dataset}_jains_index_hitrates_exp_smth_T{T}_m{num_users}.pdf',bbox_inches="tight")


def generate_plots_reward_rates():
    set_theme()
    ops_ofa, ops_ga, _=load_output()
    #### plot alpha vs total hitrate
    # ops_alg=ops_greedy
    # ops_alg=ops_ogd
    for ops_alg in [ops_ga,ops_ofa]:
        means=[]
        alphas=[]
        mins=[]
        maxs=[]
        for (k,op) in sorted(ops_alg.items(),key= lambda x:float(x[0])):
            alpha=float(k)
            # if alpha>1:
            #     break
            # s=0
            # print(f'\n{k}')
            # print('rates_optimal\t', end=' ')
            # for i in range(num_users):
            #     s+=op[k].rates_optimal[i][-1]
                # print(f'{ops_ogd[k].rates_optimal[i][-1]:.3f}', end='\t')
            # print(f'{np.min(ops_ogd[k].rates_optimal[:,-1]):.3f}',end=' ')
            # print(f'{s/num_users:.3f}')
            s=0
            # print('rates_online\t\t', end=' ')
            for i in range(num_users):
                s+=op.rates[i][-1]
                # print(f'{op.rates[i][-1]:.3f}', end='\t')
            # print(f'{np.min(op.rates[:,-1]):.3f}',end=' ')
            # print(f'{s/num_users:.3f}')
            means.append(s/num_users)
            alphas.append(alpha)
            mins.append(np.min(op.rates[:,-1]))
            maxs.append(np.max(op.rates[:,-1]))
        plt.figure()
        plt.plot(alphas,means,linewidth=2.5,label='average')
        plt.plot(alphas,maxs,linestyle='dashed',label='max')
        plt.plot(alphas,mins,linestyle='dashed',label='min')
        plt.fill_between(alphas,mins,maxs,alpha=0.1)
        plt.legend()
        plt.xlabel('$\\alpha$')
        plt.ylim((0,1.01))
        # plt.xlim([0,2])
        with np.printoptions(precision=3,suppress=True):    
            print('means\t',np.array(means))
            print('alpha\t',np.array(alphas))

            # print('maxs\t',np.array(maxs))
            # print('mins\t',np.array(mins))
            
        if ops_alg==ops_ga:
            # plt.title('GA Fair Scheduling')
            plt.ylabel('Reward Rates (GA)')
            plt.savefig(path+f'/{dataset}_avg_hitrates_vs_alpha_ga_T{T}_m{num_users}.pdf',bbox_inches="tight")
        else:
            # plt.title('OPF Fair Scheduling')
            plt.ylabel('Reward Rates (OFA)')
            plt.savefig(path+f'/{dataset}_avg_hitrates_vs_alpha_opf_T{T}_m{num_users}.pdf',bbox_inches="tight")


if __name__=="__main__":
    # dataset='mcs' #wlan
    dataset='wlan'
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
    genearate_plot_jains_index()
    generate_plots_reward_rates()
