import cvxpy as cp
import numpy as np
import scipy.optimize as sopt
from easydict import EasyDict as edict

def comp_hitrate(N,T,num_users,cache_size,requests,y_opt):
    R_opt=np.zeros(num_users)
    jains_index_opt=np.zeros(T)
    hitrates_optimal=[]
    for i in range(num_users):
        hitrates_optimal+=[np.zeros(T)]

    for t in range(T):
        g=np.zeros(N)
        for i in range(num_users):
            # if requests[i][t] in y_opt:
            R_opt[i]+=y_opt[requests[i][t]]
            hitrates_optimal[i][t]=(R_opt[i])/(t+1)
        jains_index_opt[t]=np.sum(R_opt)**2/(num_users*np.sum(np.power(R_opt,2)))

    return edict({'hitrates':np.array(hitrates_optimal), 'jains_index':jains_index_opt})

##### projection functions
def ogd_projection_fm(u,N,T,num_users,cache_size,max_iter=500):
    y_k=np.array([num_users/N]*N)

    for step in range(max_iter):
        df_y_k = (2*u*y_k-2*u*u)
        idx=df_y_k.argsort()
        # print(idx)
        z_k=np.zeros(N)
        z_k[idx[:cache_size]]=1
        # print(z_k)
        eta=(2./(step+2))
        y_prev=y_k
        y_k=y_k+eta*(z_k-y_k)
    return y_k

def ogd_projection_cvxpy(u,N,T,num_users,cache_size):
        y=cp.Variable(N,name='y')
        val=-cp.norm(y-u)
        # val=cum_surrogate_reward @ y - (1./(2*eta_t))*cp.norm(y)**2
        constr=[]
        for i in range(N):
            constr+=[y[i]>=0,y[i]<=1]
        constr+=[cp.sum(y)==cache_size]
        problem=cp.Problem(cp.Maximize(val),constr)
        problem.solve()

        ### cache configuration for next iteration
        return y.value

def ogd_projection_scipy(u,N,T,num_users,cache_size):
    fun= lambda x: np.sum(np.power(x-u,2))
    lc1=sopt.LinearConstraint(np.eye(N),0,1)
    lc2=sopt.LinearConstraint(np.ones(N),cache_size,cache_size)
    bnd=sopt.Bounds(0,1)
    x0=np.array([num_users/N]*N)
    res=sopt.minimize(fun,x0,constraints=[lc1,lc2],bounds=bnd)
    return res.x

##### Madow's Sampling
def madow_sampling(p,k):
    N = len(p)
    y = np.zeros(N)
    P = p.cumsum() ### cumulative probability
    # print('P',P)
    U=np.random.rand() ### uniform random number in [0,1]
    Us = np.arange(k)+U
    # print('Us: ', Us)
    idxs = np.searchsorted(P,Us)
    # print('idxs: ', idxs)
    y[idxs]=1
    return y
