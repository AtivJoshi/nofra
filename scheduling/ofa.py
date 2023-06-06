import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from tools import madow_sampling, ogd_projection_scipy

def ofa(T,num_users,requests,alpha,y_opt):
    
    R_opt=np.ones(num_users)
    R=np.ones(num_users)
    R_sampled=np.ones(num_users)

    rates_optimal=np.zeros((num_users,T))
    rates=np.zeros((num_users,T))
    rates_sampled=np.zeros((num_users,T))
    
    cum_sq_reward=0

    y_t=np.array([1/num_users]*num_users)
    cum_surrogate_reward=np.zeros(num_users)

    regret=np.zeros(T)
    regret_sampled=np.zeros(T)
    c_regret=np.zeros(T)
    jains_index=np.zeros(T)
    phi_online=np.zeros(T)
    phi_opt=np.zeros(T)
    
    u_t=np.zeros(num_users)

    for t in tqdm(range(1,T)):

        #### update the rewards for current configuration
        R+=y_t*requests[:,t]
        rates[:,t]=(R-1)/(t+1)

        R_opt+=y_opt*requests[:,t]
        rates_optimal[:,t]=(R_opt-1)/(t+1)

        y_sampled = madow_sampling(y_t,1)
        R_sampled += y_sampled*requests[:,t]
        rates_sampled[:,t]=(R_sampled-1)/(t+1)

        #### update metrics
        phi_opt[t]=np.sum(np.power(R_opt,1-alpha))
        phi_online[t]=np.sum(np.power(R,1-alpha))
        regret[t]= phi_opt[t] - phi_online[t]
        # print(R_sampled)
        regret_sampled[t]= np.sum(np.power(R_opt,1-alpha)) - np.sum(np.power(R_sampled,1-alpha))
        c_regret[t]= np.sum(np.power(R_opt,1-alpha)) - 1.445*np.sum(np.power(R,1-alpha))
        jains_index[t]=np.sum(R)**2/(num_users*np.sum(np.power(R,2)))

        #### compute the next configuration
        g = np.zeros(num_users)
        for i in range(num_users):
            g[i]+=requests[i,t]/np.power(R[i],alpha)
        cum_surrogate_reward+=g

        ### optimal eta
        cum_sq_reward+=np.sum(np.square(g))
        # D=np.sqrt(2*num_users)
        # eta_t=np.sqrt(2)*D/(2*np.sqrt(cum_sq_reward))
        eta_t=np.sqrt(2)/np.sqrt(cum_sq_reward)
        # eta_t=0.01

        ### projection step
        ### cache configuration for next iteration
        u_t=y_t+eta_t*g
        y_t=ogd_projection_scipy(u_t,T,num_users)
    
    return edict({'rates':np.array(rates), 'rates_optimal':np.array(rates_optimal), 
                  'regret':regret, 'c_regret':c_regret, 'regret_sampled':regret_sampled, 
                  'jains_index':jains_index,'rates_sampled':rates_sampled,
                  'phi_opt':phi_opt, 'phi_online':phi_online, 'R':R, 'R_opt':R_opt,
                  'R_sampled':R_sampled})
    