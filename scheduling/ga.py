import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

def ga(T,num_users,requests,alpha,y_opt,beta=0.001):
    R_exp_smth=np.ones(num_users)
    R=np.ones(num_users)
    R_opt=np.ones(num_users)
    rates=np.zeros((num_users,T))
    rates_optimal=np.zeros((num_users,T))
    phi_online=np.zeros(T)
    # u_avg=requests.cumsum(axis=1)/np.arange(1,T+1)
    regret=np.zeros(T)
    jains_index=np.zeros(T)
    for t in tqdm(range(1,T)):
        scheduler=np.power(R_exp_smth,-alpha)*requests[:,t-1]
        y_t_idx=scheduler.argmax()
        R_exp_smth=(1-beta)*R_exp_smth
        R_exp_smth[y_t_idx]+=beta*requests[y_t_idx,t]
        R[y_t_idx]+=requests[y_t_idx,t]
        rates[:,t]=(R-1)/(t+1)

        R_opt+=y_opt*requests[:,t]
        rates_optimal[:,t]=(R_opt-1)/(t+1)
        phi_online[t]=np.sum(np.power(R,1-alpha))

        regret[t]= np.sum(np.power(R_opt,1-alpha)) - np.sum(np.power(R,1-alpha))
        jains_index[t]=np.sum(R)**2/(num_users*np.sum(np.power(R,2)))

    return edict({'rates':rates, 'jains_index':jains_index,'regret':regret,
                  'phi_online':phi_online, 'R':R, 'R_opt':R_opt})