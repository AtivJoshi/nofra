import numpy as np
import scipy.optimize as sopt

def ogd_projection_scipy(u,T,num_users):
    fun= lambda x: np.sum(np.power(x-u,2))
    lc1=sopt.LinearConstraint(np.eye(num_users),0,1)
    lc2=sopt.LinearConstraint(np.ones(num_users),1,1)
    bnd=sopt.Bounds(0,1)
    x0=np.array([1./num_users]*num_users)
    res=sopt.minimize(fun,x0,constraints=[lc1,lc2],bounds=bnd)
    return res.x

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