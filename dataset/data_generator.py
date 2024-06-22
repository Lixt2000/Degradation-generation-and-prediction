import numpy as np
import torch
from numpy import ndarray, dtype, random
from scipy.stats import geninvgauss

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(2023)

def inverse_gaussian_process(Time, N, alpha, beta):
    
    dt = Time/N
    
    inverse_gaussian_values = geninvgauss.rvs(alpha*dt, beta*dt**2, size=N) / 50
    inverse_gaussian_process = np.cumsum(inverse_gaussian_values)
    inverse_gaussian_process = np.insert(inverse_gaussian_process, 0, 0.0)
    return inverse_gaussian_process
    
Time = 20
N = 20
sample_num = 20#00
alpha0 = 2.0
beta0 = 1.0
ori_data = []

sample_cnt = 0
for i in range(sample_num):
    ori_data.append(inverse_gaussian_process(Time, N, alpha0, beta0))
    sample_cnt += 1
    if sample_cnt % 10 == 0:
        print("generate {} total {}".format(sample_cnt, sample_num))
ori_data = np.array(ori_data)

save_path = '../data/test_data.npy'
np.save(save_path, ori_data)
print('generate data done!')
# data = np.load(save_path)
