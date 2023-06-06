import numpy as np

def optimal_config(T,num_users,requests,alpha,max_iter=2000):

    #### compute the cumulative file requests
    X_T = requests.sum(axis=1)

    #### initialize y_k
    y_k=np.array([1./num_users]*num_users)

    for step in range(max_iter):
        #### compute the gradient at y_k
        nabla_f_y_k = X_T**(1-alpha)*y_k**(-alpha)
        # print(nabla_f_y_k,nabla_f_y_k.argmax())
        z_k=np.zeros(num_users)
        z_k[nabla_f_y_k.argmax()]=1
        eta=(2./(step+3))
        # y_prev=y_k
        y_k=y_k+eta*(z_k-y_k)

    return y_k, X_T@y_k

# y_opt=optimal_config(T,num_users,requests,alpha,max_iter=2000)