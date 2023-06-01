import copy
import os.path
import pickle

import cvxpy as cp
import h5py
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as sopt
import seaborn as sns
from easydict import EasyDict as edict
from scipy.special import entr
from tqdm import tqdm
from memoization import cached, CachingAlgorithmFlag

def generate_data(N,T,num_users,dists=None,rng=None):
    '''
    N: library size
    T: horizon
    num_users: number of users
    dists: list of probability distribution for each user
    rng: numpy random number generator object
    '''

    if rng is None:
        rng=np.random.default_rng()
    # list of probability distribution for each user
    requests=[] # list of generated requests 
    if dists is None:
        dists=[]
        for u in range(num_users): 
            q=rng.random(N)
            # q=np.ones(N)
            p=q/np.sum(q)
            dists.append(p)
    elif dists=='uniform':
        dists=[]
        for u in range(num_users): 
            q=np.ones(N)
            p=q/np.sum(q)
            dists.append(p)

    for u in range(num_users): 
        requests.append(rng.choice(N,T,p=dists[u]))
        # requests.append(rng.choice(N,T))
    return dists, requests


def load_cmu_data(num_users:int, num_files:int,folder_path:str="",file_name="CMU_huge")->np.ndarray:
    file_path=folder_path+f'/{file_name}.txt'
    cache_path=folder_path+f'/{file_name}_{num_users}u_{num_files}f_cache.npy'

    if os.path.isfile(cache_path):
        return np.load(cache_path)
    else:
        df = pd.read_csv(file_path, sep = ' ',engine='python')
        df.columns = ['Req_ID', 'File_ID', 'File_Size']
        # To control the size of the library, we can rename the file i to (i % num_files). 
        # This results in extremely bad accuracy, so avoiding it. Instead, drop the files when file_name > num_files.
        old_id = df.File_ID.unique()
        old_id.sort()
        new_id = dict(zip(old_id, np.arange(len(old_id))))
        df = df.replace({"File_ID": new_id})
        df.drop(list(df[df['File_ID']>=num_files].index),inplace=True) ##pyright: reportGeneralTypeIssues=false

        # array of file requests
        raw_seq=df['File_ID'].to_numpy()

        # split raw_seq into chunks of size <num_users>
        num_requests=raw_seq.size//num_users
        input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_users))
        np.save(cache_path,input_seq.T)
        return input_seq.T 

def load_cmu_data1(num_users:int, num_files:int,folder_path:str="",file_name="CMU_huge")->np.ndarray:
    #num_files set to 50, num_users to 3
    num_files=50
    num_users=3
    file_path=folder_path+f'/{file_name}.txt'
    cache_path=folder_path+f'/{file_name}_{num_users}u_{num_files}f_cache.npy'

    if os.path.isfile(cache_path):
        return np.load(cache_path)
    else:
        df = pd.read_csv(file_path, sep = ' ',engine='python')
        df.columns = ['Req_ID', 'File_ID', 'File_Size']
        # To control the size of the library, we can rename the file i to (i % num_files). 
        # This results in extremely bad accuracy, so avoiding it. Instead, drop the files when file_name > num_files.
        old_id = df.File_ID.unique()
        old_id.sort()
        new_id = dict(zip(old_id, np.arange(len(old_id))))
        df = df.replace({"File_ID": new_id})
        df.drop(list(df[df['File_ID']>=num_files].index),inplace=True) ##pyright: reportGeneralTypeIssues=false

        file_id=df['File_ID'].to_numpy()
        input_seq=file_id[:1500] #select first 1500 requests
        f,f_count=np.unique(input_seq,return_counts=True)
        count_sort_ind=np.argsort(-f_count) #sort files by their counts


        #relabel files according to the counts in descending order

        input_seq1=np.zeros(1500)
        f1=f[count_sort_ind]
        for i in range(num_files):
            input_seq1[np.argwhere(input_seq==f1[i])]=i
        
        requests=np.zeros((3,500))
        counts=np.zeros(3,dtype=int)
        for i in range(1500):
            if input_seq1[i]<=5 and counts[0]<500:
                requests[0,counts[0]]=input_seq1[i]
                counts[0]+=1
            elif (input_seq1[i]<=10) and (counts[1]<500):
                requests[1,counts[1]]=input_seq1[i]
                counts[1]+=1
            else:
                requests[2,counts[2]]=input_seq1[i]
                counts[2]+=1
        
        requests=requests.astype(int)
        np.save(cache_path,requests.T)
        return requests.T

        # array of file requests
        # raw_seq=df['File_ID'].to_numpy()

        # # split raw_seq into chunks of size <num_users>
        # num_requests=raw_seq.size//num_users
        # input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_users))
        # #np.save(cache_path,input_seq.T)
        # return input_seq.T 

def load_cmu_data2(num_users:int, num_files:int,folder_path:str="",file_name="CMU_huge")->np.ndarray:
    #num_files set to 50, num_users to 4
    num_files=50
    num_users=4
    file_path=folder_path+f'/{file_name}.txt'
    cache_path=folder_path+f'/{file_name}_{num_users}u_{num_files}f_cache.npy'

    if os.path.isfile(cache_path):
        return np.load(cache_path)
    else:
        df = pd.read_csv(file_path, sep = ' ',engine='python')
        df.columns = ['Req_ID', 'File_ID', 'File_Size']
        # To control the size of the library, we can rename the file i to (i % num_files). 
        # This results in extremely bad accuracy, so avoiding it. Instead, drop the files when file_name > num_files.
        old_id = df.File_ID.unique()
        old_id.sort()
        new_id = dict(zip(old_id, np.arange(len(old_id))))
        df = df.replace({"File_ID": new_id})
        df.drop(list(df[df['File_ID']>=num_files].index),inplace=True) ##pyright: reportGeneralTypeIssues=false

        file_id=df['File_ID'].to_numpy()
        input_seq=file_id[:1600] #select first 1600 requests
        f,f_count=np.unique(input_seq,return_counts=True)
        count_sort_ind=np.argsort(-f_count) #sort files by their counts


        #relabel files according to the counts in descending order

        input_seq1=np.zeros(1600)
        f1=f[count_sort_ind]
        for i in range(num_files):
            input_seq1[np.argwhere(input_seq==f1[i])]=i
        
        requests=np.zeros((4,400))
        counts=np.zeros(4,dtype=int)
        for i in range(1600):
          if input_seq1[i]<=5 and counts[0]<400:
              requests[0,counts[0]]=input_seq1[i]
              counts[0]+=1
          elif input_seq1[i]<=8 and counts[1]<400:
              requests[1,counts[1]]=input_seq1[i]
              counts[1]+=1
          elif input_seq1[i]<=11 and counts[2]<400:
              requests[2,counts[2]]=input_seq1[i]
              counts[2]+=1
          else:
              requests[3,counts[3]]=input_seq1[i]
              counts[3]+=1
        
        requests=requests.astype(int)
        np.save(cache_path,requests.T)
        return requests.T

        # array of file requests
        # raw_seq=df['File_ID'].to_numpy()

        # # split raw_seq into chunks of size <num_users>
        # num_requests=raw_seq.size//num_users
        # input_seq=np.array(np.array_split(raw_seq[:num_users*num_requests],num_users))
        # #np.save(cache_path,input_seq.T)
        # return input_seq.T 
