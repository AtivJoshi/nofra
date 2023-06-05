import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
from tools import ogd_projection_cvxpy, ogd_projection_fm, ogd_projection_scipy

def si_salem_algo(N,T,num_users,cache_size,requests,alpha,y_opt,projection_func='scipy'):

    if projection_func == 'scipy':
        projection=ogd_projection_scipy
    elif projection_func == 'cvxpy':
        projection=ogd_projection_cvxpy
    elif projection_func == 'fw':
        projection=ogd_projection_fm

    if alpha==0:
        alpha=0.01

    #### check if alphas are different for each user
    # if isinstance(alpha,(float,int)):
    #     a=np.zeros(num_users)
    #     a[:]=float(alpha)
    #     alpha=a

    u_min=1./T #do we need the assumption that ||
    u_max=1

    R_opt=np.ones(num_users)
    R=np.ones(num_users)

    hitrates_optimal=[]
    hitrates=[]
    for i in range(num_users):
        hitrates.append(np.zeros(T))
        hitrates_optimal+=[np.zeros(T)]
        #hitrates_sampled+=[np.zeros(T)]

    cum_sq_reward=0

    y_t=np.array([cache_size/N]*N)
    cum_surrogate_reward=np.zeros(N)

    # theta_t=np.array([0.5*((-1.0/u_min)+(-1.0/u_max))]*num_users)

    theta_t = np.ones(num_users)

    regret=np.zeros(T)
    c_regret=np.zeros(T)
    phi_diff_sampled=np.zeros(T)
    jains_index=np.zeros(T)
    downloads=np.zeros(T)

    online_reward=0
    optimal_reward=0

    u_t=np.zeros(N)
    utility_t=np.zeros(num_users)
    for t in tqdm(range(T)):
        print(y_t)
        grad_x=np.zeros(N)

        ### "play" the current configuration
        ### update the fractional hits at time t
        for i in range(num_users):
            r=np.zeros(N)
            utility_t[i]=y_t[requests[i][t]]
            r[requests[i][t]]=1
            grad_x += theta_t[i]*r
            R[i]+=y_t[requests[i][t]]
            hitrates[i][t]=(R[i]-1)/(t+1)

            R_opt[i]+=y_opt[requests[i][t]]
            hitrates_optimal[i][t]=(R_opt[i]-1)/(t+1)

            #y_sampled = madow_sampling(y_t,cache_size)
            #R_sampled[i]+=y_sampled[requests[i][t]]
            #hitrates_sampled[i][t]=(R_sampled[i]-1)/(t+1)

        # if alpha<1:
        #   conjugate_f=np.sum(((alpha*np.power(-theta_t,1-(1/alpha)))-1)/(1-alpha))
        # else:
        #   conjugate_f=np.sum(-np.log(-theta_t)-1)

        # zhi=conjugate_f-np.dot(theta_t,utility_t)

        cum_sq_reward+=np.sum(np.square(grad_x))

        grad_theta=np.power(-theta_t,-(1/alpha))-utility_t
        eta_x=np.sqrt(2*cache_size)/np.sqrt(cum_sq_reward)
        eta_theta=(alpha*np.power(u_min,-1-(1/alpha)))/(t+1)

        regret[t]= np.sum(np.power(R_opt,1-alpha)) - np.sum(np.power(R,1-alpha))
        c_regret[t]= np.sum(np.power(R_opt,1-alpha)) - 1.445*np.sum(np.power(R,1-alpha))
        #phi_diff_sampled[t] = np.sum(np.power(R,1-alpha)) - np.sum(np.power(R_sampled,1-alpha))
        jains_index[t]=np.sum(R)**2/(num_users*np.sum(np.power(R,2)))

        ### compute the next configuration
        ### compute surrogate reward using the fractional hits


        ### projection step
        ### cache configuration for next iteration
        u_t=y_t+(eta_x*grad_x)
        w_t=theta_t-(eta_theta*grad_theta)
        # y_t=y.value
        # y_t=ogd_projection(u_t,N,T,num_users,cache_size)
        # y_t=ogd_projection_scipy(u_t,N,T,num_users,cache_size)
        y_t_prev=y_t
        y_t = projection(u_t,N,T,num_users,cache_size)

        #projection step for theta:
        for i in range(num_users):
            if w_t[i]<-(1/np.power(u_min,alpha)):
                theta_t[i]=-(1/u_min)
            elif w_t[i]>-(1/u_max):
                theta_t[i]=-(1/u_max)
            else:
                theta_t[i]=w_t[i]

        #theta_t = projection(w_t,N,T,num_users,cache_size)

        difference=y_t - y_t_prev
        downloads[t]=np.sum(difference[difference>0])

    return edict({'hitrates':np.array(hitrates), 'hitrates_optimal':np.array(hitrates_optimal),
                  'regret':regret, 'c_regret':c_regret, 'jains_index':jains_index,
                  'phi_diff_sampled':phi_diff_sampled,
                  'downloads':downloads})


def si_salem_algo2(N,T,num_users,cache_size,requests,alpha,y_opt):
    u_min,u_max = (1./T,1.)
    # theta_min = -1./np.power(u_min,alpha)
    # theta_max = -1./np.power(u_max,alpha)

    ## as the algorithm cannot handle alpha=0
    if alpha==0:
        alpha=0.01

    y_t=np.array([cache_size/N]*N)
    theta_t = np.ones(num_users)

    cum_norm_sq_g_x=0

    R=np.ones(num_users)
    jains_index=np.zeros(T)
    hitrates=[]
    for i in range(num_users):
        hitrates.append(np.zeros(T))

    for t in tqdm(range(T)):
        # print(y_t)
        # compute gradient w.r.t. x

        for i in range(num_users):
            R[i]+=y_t[requests[i][t]]
            hitrates[i][t]=(R[i]-1)/(t+1)

        jains_index[t]=np.sum(R)**2/(num_users*np.sum(np.power(R,2)))


        g_x = np.zeros(N)
        for i in range(num_users):
            g_x[requests[i][t]]+=theta_t[i]

        # compute gradient w.r.t. theta (as per Si Salem's Github code, 
        # the update step in the paper is different)
        g_theta = np.zeros(num_users)
        for i in range(num_users):
            # print(y_t[requests[i][t]])
            g_theta[i]=-1./(theta_t[i])**(1/alpha) + y_t[requests[i][t]]

        # compute next cache configuration by projection
        cum_norm_sq_g_x+=np.sum(np.square(g_x))
        eta_x_t = np.sqrt(2*num_users)/np.sqrt(cum_norm_sq_g_x)
        u = y_t + eta_x_t*g_x
        y_t = ogd_projection_scipy(u,N,T,num_users,cache_size)

        # compute next theta by projection
        eta_theta = alpha/(u_min**(1 + 1./alpha)*(t+1))
        theta_t = theta_t - eta_theta*g_theta
        theta_t[theta_t > 1. / u_min ** alpha] = 1./u_min ** alpha
        theta_t[theta_t < 1. / u_max ** alpha] = 1./u_max ** alpha

    return edict({'hitrates':np.array(hitrates), 'jains_index':jains_index})