import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats.qmc as sq
from numpy import ndarray, dtype, random
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_s_curve, make_moons
import torch
import geomloss
from torch import nn, optim
from DDPM2d import ddpm1d, p_sample_loop, q_x, p_sample_loop_V
from DCGAN import train, discriminator, generator
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from increment import increment, sample_extend0, sample_extend
from main_timegan import main_timegan

from prediction import prediction
from scipy.stats import gamma, geninvgauss, invgauss, entropy, wasserstein_distance, norm
from scipy.special import digamma
from scipy import optimize
import seaborn as sns
from svaegrutrain import svae_gru
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_random_points_in_ring(r_in, r_out, n_points):
    points = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(r_in, r_out)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        points.append((x, y))

    return points


def add_index(data):
    result = []
    for i, value in enumerate(data, start=1):
        result.append([value])
    return result


def wiener_process(Time, N, mu=0, sigma=1):
    """
    Generate a sample of the Wiener process.
    Time(float): Total time interval.
    N(int): Number of timr steps.
    mu(float): Mean of the Wiener process.
    sigma(float): Standard deviation of the Wiener process.

    Returns:
        numpy.ndarray: Array of shape (N+1,).
    """
    dt = Time/N
    dW = norm.rvs(loc=mu * dt, scale=sigma * dt, size=N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0.0)
    return W


def gamma_process(Time, N, alpha, beta):
    """
    Generate a sample of the Gamma process.
    Time(float): Total time interval.
    N(int): Number of time steps.
    alpha(float): shape parameter of the Gamma process.
    beta(float): rate parameter of the Gamma process.

    Returns:
        numpy.ndarray: Array of shape (N+1,).
    """
    dt = Time/N
    gamma_values = gamma.rvs(alpha, scale=1/beta, size=N)#  * dt
    gamma_process = np.cumsum(gamma_values)
    gamma_process = np.insert(gamma_process, 0, 0.0)
    return gamma_process


def inverse_gaussian_process(Time, N, alpha, beta):
    """
    Generate a sample of the Inverse Gaussian process.
    Time(float): Total time interval.
    N(int): Number of time steps.
    alpha(float): shape parameter of the Inverse Gaussian process.
    beta(float): scale parameter of the Inverse Gaussian process.

    Returns:
        numpy.ndarray: Array of shape (N+1,).
    """
    dt = Time/N
    # u_values = np.random.uniform(0, 1, size=N)
    # inverse_gaussian_values = beta * (1 + (beta**2 * u_values**2 - 2 * alpha * u_values)**0.5)
    inverse_gaussian_values = invgauss.rvs(mu=alpha*dt, scale=beta*dt**2, size=N)
    inverse_gaussian_process = np.cumsum(inverse_gaussian_values)
    inverse_gaussian_process = np.insert(inverse_gaussian_process, 0, 0.0)
    return inverse_gaussian_process


setup_seed(2023)

D = 10 ** 2

data_name = 'Inverse Gaussian' # 'wiener', 'gamma', 'Inverse Gaussian', 'Fatigue', 'Train wheel', 'Laser', 'Interial navigition'

Time = 10 #10#20
N = 10 #10#20
sample_num = 10 #10#20 #15#30

ori_data = []

#x = np.linspace(0, 0.1, 11)
#x = np.linspace(2.5, 22.5, 9)
#x = np.linspace(0, 1.0, 21)

if data_name == 'gamma':
    standard_process = gamma_process
    alpha0 = 2
    beta0 = 1
    for i in range(sample_num):
        ori_data.append(standard_process(Time, N, alpha0, beta0))  # / 50
    ori_data = np.array(ori_data)
    constant0 = ori_data[:, 0]
    constant = ori_data[:int(sample_num * 0.8), 0] #int(sample_num * 0.8)
    train_data = ori_data[:int(sample_num * 0.8), :]
    test_data = ori_data[int(sample_num * 0.8):, :]

    batch_size = 32
    num_steps = 1400 #200 #1000 #1400
    num_epoch = 90000 #60000 #120000 #80000 #100000
    t_epoch = 3000
    t_batch = 32
    threshold = 1.8
    time_limit = 18
    svae_epoch = 200
    svae_path = 'output/gamma10_21_200/epoch_181_loss_0.08103876560926437.pt'
    #'output/gamma10_11_200/epoch_195_loss_0.09454663842916489.pt'
    #'output/gamma20_11_200/epoch_177_loss_0.16919507086277008.pt'
    #'output/gamma10_21_200/epoch_181_loss_0.08103876560926437.pt'
    #'output/gamma20_21_200/epoch_178_loss_0.17532505095005035.pt'

    svae_state = 'predict'  # 'train'
    lengths = N - 1  # 4
    BATCH_SIZE = 1  # 2
    D_pred = 5
    x = np.linspace(0, N, N+1)

elif data_name == 'Inverse Gaussian':
    standard_process = inverse_gaussian_process
    alpha0 = 0.1
    beta0 = 1
    for i in range(sample_num):
        ori_data.append(standard_process(Time, N, alpha0, beta0))  # / 50
    ori_data = np.array(ori_data)
    constant0 = ori_data[:, 0]
    constant = ori_data[:int(sample_num * 0.8), 0] #int(sample_num * 0.8)
    train_data = ori_data[:int(sample_num * 0.8), :]
    test_data = ori_data[int(sample_num * 0.8):, :]

    batch_size = 32
    num_steps = 200 #200 #500
    num_epoch = 30000 #30000 #55000 #88000 #190000   #50000 #100000 #150000 #250000
    t_epoch = 3000
    t_batch = 32
    threshold = 0.8 #0.8 #1.8
    time_limit = 8 #8 #18
    svae_epoch = 200
    svae_path = 'output/Inverse Gaussian10_11_200/epoch_136_loss_0.01136336475610733.pt'
    #'output/Inverse Gaussian10_11_200/epoch_136_loss_0.01136336475610733.pt'
    #'output/Inverse Gaussian10_21_200/epoch_180_loss_0.014063547365367413.pt'
    #'output/Inverse Gaussian20_11_200/epoch_152_loss_0.024518223479390144.pt'
    #'output/Inverse Gaussian20_21_200/epoch_194_loss_0.027105828747153282.pt'

    #'output/Inverse Gaussian30_21_300/epoch_197_loss_0.045035687275230885.pt'
    #'output/Inverse Gaussian15_21_300/epoch_295_loss_0.050799574702978134.pt'
    #'output/Inverse Gaussian30_11_300/epoch_195_loss_0.044036814011633396.pt'
    svae_state = 'predict' #'train'
    lengths = N-1#4
    BATCH_SIZE = 1#2
    D_pred = 5
    x = np.linspace(0, N, N+1)

elif data_name == 'wiener':
    standard_process = wiener_process
    alpha0 = 0
    beta0 = 1
    for i in range(sample_num):
        ori_data.append(standard_process(Time, N, alpha0, beta0))  # / 50
    ori_data = np.array(ori_data)
    constant0 = ori_data[:, 0]
    constant = ori_data[:int(sample_num * 0.8), 0] #int(sample_num * 0.8)
    train_data = ori_data[:int(sample_num * 0.8), :]
    test_data = ori_data[int(sample_num * 0.8):, :]

    batch_size = 32
    num_steps = 200 #200 #500
    num_epoch = 170000  #150000 #170000 #120000 #160000
    t_epoch = 3000
    t_batch = 32
    threshold = 0.8
    time_limit = 8
    svae_epoch = 200
    svae_path = 'output/wiener10_11_200/epoch_195_loss_0.05284277722239494.pt'
    #'output/wiener10_11_200/epoch_195_loss_0.05284277722239494.pt'
    # 'output/wiener20_11_200/epoch_166_loss_0.10882940143346786.pt'
    # 'output/wiener10_21_200/epoch_195_loss_0.046819597482681274.pt'
    #'output/wiener20_21_200/epoch_171_loss_0.1074134036898613.pt'

    svae_state = 'train'  # 'train'
    lengths = N - 1  # 4
    BATCH_SIZE = 1  # 2
    D_pred = 5
    x = np.linspace(0, N, N+1)

elif data_name == 'Fatigue':
    ori_data0 = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/fatigue.xlsx', header=0,
                             index_col=0)
    ori_data = ori_data0.to_numpy()[1:, :11]
    constant0 = ori_data[:, 0]
    constant = ori_data[:16, 0] #16
    train_data = ori_data[:16, :11]
    test_data = ori_data[16:, :11]

    batch_size = 32
    num_steps = 100
    num_epoch = 80000 #80000
    t_epoch = 5000
    t_batch = 16
    threshold = 1.35
    time_limit = 10
    svae_epoch = 300 #300 #200
    svae_path = 'output/Fatigue20_11_300/epoch_186_loss_0.014251934364438057.pt'
    # 'output/Fatigue20_11_200/epoch_170_loss_0.016254637390375137.pt'
    # 'output/Fatigue20_11_300/epoch_186_loss_0.014251934364438057.pt'
    svae_state = 'predict'  # 'train' 'predict'
    lengths = 9
    BATCH_SIZE = 1
    D_pred = 5
    x = np.linspace(0, 0.1, 11)

elif data_name == 'Train wheel':
    ori_data0 = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/trainwheel.xlsx', header=0)
    ori_data = ori_data0.to_numpy()[:11, :]
    constant0 = ori_data[:, 0]
    constant = ori_data[:8, 0] #8
    train_data = ori_data[:8, :]
    test_data = ori_data[8:, :]

    batch_size = 32
    num_steps = 1800
    num_epoch = 140000 #80000
    t_epoch = 5000
    t_batch = 16
    threshold = 50
    time_limit = 12
    svae_epoch = 200 #200
    svae_path = 'output/Train wheel11_13_200/epoch_197_loss_0.09715424478054047.pt'
    #'output/Train wheel10_13_200/epoch_161_loss_0.5446860790252686.pt'
    svae_state = 'predict'  # 'train' 'predict'
    lengths = 11
    BATCH_SIZE = 1
    D_pred = 5
    x = np.linspace(0, 600, 13)

elif data_name == 'Laser':
    ori_data0 = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/laserdata.xlsx', header=0)
    ori_data = ori_data0.to_numpy()[:15, :]
    constant0 = ori_data[:, 0]
    constant = ori_data[:12, 0] #12
    train_data = ori_data[:12, :]
    test_data = ori_data[12:, :]

    batch_size = 32
    num_steps = 800
    num_epoch = 90000#80000
    t_epoch = 5000
    t_batch = 16
    threshold = 10
    time_limit = 15
    svae_epoch = 200
    svae_path = 'output/Laser15_17_200/epoch_195_loss_0.05121845193207264.pt'
    svae_state = 'predict'  # 'train'
    lengths = 15#4
    BATCH_SIZE = 1
    D_pred = 5
    x = np.linspace(0, 4000, 17)

elif data_name == 'Interial navigition':
    ori_data0 = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/interial navigition.xlsx',
                             header=0, index_col=0)
    ori_data = ori_data0.to_numpy()[:, :9]
    constant0 = ori_data[:, 0]
    constant = ori_data[:3, 0] #3
    train_data = ori_data[:3, :9]
    test_data = ori_data[3:, :9]

    batch_size = 32
    num_steps = 100
    num_epoch = 130000 #150000
    t_epoch = 5000
    t_batch = 5
    threshold = 0.5
    time_limit = 9
    svae_epoch = 200
    svae_path = 'output/Inverse Gaussian10_11_200/epoch_192_loss_0.018531057983636856.pt'
    svae_state = 'train'  # 'train'
    lengths = 7
    BATCH_SIZE = 1
    D_pred = 5
    x = np.linspace(2.5, 22.5, 9)

if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
    NN = len(x)
    fig, ax = plt.subplots()
    x = np.linspace(0, NN - 1, NN)
    ax.plot(x, np.array(ori_data).T, 'o--')
    plt.xlabel("Time")
    plt.ylabel(data_name + ' degradation value')
    # Save figures
    if not os.path.exists(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/'):
        os.makedirs(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/real.png')
    plt.close()

    indices = [i for i in range(1, NN)]
    incre_data = increment(ori_data, indices)
    fig, ax = plt.subplots()
    x = np.linspace(0, NN - 2, NN - 1)
    ax.plot(x, np.array(incre_data).T, 'o--')
    plt.xlabel("Time")
    plt.ylabel('Increment of ' + data_name + ' degradation value')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/increment.png')
    plt.close()

elif data_name == 'Fatigue':
    fig, ax = plt.subplots()
    x = np.linspace(0, 0.12, 13)
    ax.plot(x, np.array(ori_data0).T, 'o--')
    #x_ticks = np.arange(0, .13, 0.01)
    #plt.xticks(x_ticks)
    plt.xlabel("Million cycles")
    plt.ylabel('Length of crack')
    if not os.path.exists(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/'):
        os.makedirs(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/real.png')
    plt.close()

    #ori_data = ori_data0.to_numpy()[1:, :11]
    indices = [i for i in range(1, ori_data.shape[1])]
    incre_data = increment(ori_data, indices)
    fig, ax = plt.subplots()
    x = np.linspace(0, 9, 10)
    ax.plot(x, np.array(incre_data).T, 'o--')
    plt.xlabel("Million cycles")
    plt.ylabel('Increment Length of crack')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/increment.png')
    plt.close()

elif data_name == 'Train wheel':
    fig, ax = plt.subplots()
    x = np.linspace(0, 600, 13)
    ax.plot(x, np.array(ori_data0).T, 'o--')
    #x_ticks = np.arange(0, 650, 50)
    #plt.xticks(x_ticks)
    plt.ylabel('Deg(mm)')
    plt.xlabel("Dis(1e4 km)")
    if not os.path.exists(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/'):
        os.makedirs(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/real.png')
    plt.close()

    #ori_data = ori_data0.to_numpy()[:10, :]
    indices = [i for i in range(1, ori_data.shape[1])]
    incre_data = increment(ori_data, indices)
    fig, ax = plt.subplots()
    x = np.linspace(0, 550, 12)
    ax.plot(x, np.array(incre_data).T, 'o--')
    plt.ylabel('Deg(mm)')
    plt.xlabel("Dis(1e4 km)")
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/increment.png')
    plt.close()

elif data_name == 'Laser':
    fig, ax = plt.subplots()
    x = np.linspace(0, 4000, 17)
    ax.plot(x, np.array(ori_data0).T, 'o--')
    #x_ticks = np.arange(0, 4250, 250)
    # plt.xticks(x_ticks)
    plt.ylabel('Electric current')
    plt.xlabel("Time")
    if not os.path.exists(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/'):
        os.makedirs(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/real.png')
    plt.close()

    #ori_data = ori_data0.to_numpy()[:, :]
    indices = [i for i in range(1, ori_data.shape[1])]
    incre_data = increment(ori_data, indices)
    fig, ax = plt.subplots()
    x = np.linspace(0, 3750, 16)
    ax.plot(x, np.array(incre_data).T, 'o--')
    plt.ylabel('Electric current')
    plt.xlabel("Time")
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/increment.png')
    plt.close()

elif data_name == 'Interial navigition':
    fig, ax = plt.subplots()
    x = np.linspace(2.5, 22.5, 9)
    ax.plot(x, np.array(ori_data0).T, 'o--')
    #x_ticks = np.arange(2.5, 25, 2.5)
    #plt.xticks(x_ticks)
    plt.xlabel("Time")
    plt.ylabel('Gyroscopic drift')
    if not os.path.exists(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/'):
        os.makedirs(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/real.png')
    plt.close()

    #ori_data = ori_data0.to_numpy()[:, :9]
    indices = [i for i in range(1, ori_data.shape[1])]
    incre_data = increment(ori_data, indices)
    fig, ax = plt.subplots()
    x = np.linspace(0, 7, 8)
    ax.plot(x, np.array(incre_data).T, 'o--')
    plt.xlabel("Time")
    plt.ylabel('Increment Gyroscopic drift')
    fig.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/increment.png')
    plt.close()


if __name__ == '__main__':
    print("Number of sample:", D)
    # print(train_data)
    print("Shape of train data:", np.shape(train_data))

    indices = [i for i in range(1, train_data.shape[1])]
    train_data = increment(train_data, indices)
    train_data0 = increment(ori_data, indices)
    print("Shape of increment train data:", np.shape(train_data))

    dataset = torch.Tensor(train_data).float().cuda()
    dataset0 = torch.Tensor(train_data0).float().cuda()
    n = dataset.shape[0]
    d = dataset.shape[1]

    # 生成截断数据
    since_ss = time.time()
    expand_data0 = sample_extend(train_data0, D * sample_num, constant0)  # n
    expand_data0 = np.array(expand_data0).reshape(D, -1, d + 1)
    expand_data = sample_extend(train_data, D * n, constant)  # n
    expand_data = np.array(expand_data).reshape(D, -1, d + 1)
    print(expand_data.shape)
    print('Increment and expansion ends!')
    time_elapsed_ss = time.time() - since_ss

    # DDPM
    since_ddpm = time.time()
    G, betas, one_minus_alphas_bar_sqrt = ddpm1d(dataset, num_steps, batch_size, num_epoch, constant, d, data_name)


    def PASS_DDPM(D, dataset, data_name):
        Z_list = []
        sample_list = []
        candidate_list = []
        numb = 0
        #PASS生成D个样本
        while numb < D:
            # 初始分布抽样
            U = torch.randn(dataset.shape)

            # 使用DDPM生成
            z_candidate_seq = p_sample_loop_V(U, G, U.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
            z_candidate = z_candidate_seq[num_steps].detach()

            # 输入PASS样本
            if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
                ori_data = []
                for i in range(sample_num):
                    ori_data.append(standard_process(Time, N, alpha0, beta0))  # / 50
                ori_data = np.array(ori_data)
                # ori_data = scaler.fit_transform(ori_data.T).T
                #constant = ori_data[:int(sample_num / 2), 0]
                #train_data = ori_data[:int(sample_num / 2), :]
                #pass_data = ori_data[int(sample_num / 2):, :]
                constant = ori_data[:int(sample_num * 0.8), 0]
                train_data = ori_data[:int(sample_num * 0.8), :]
                constant0 = ori_data[:, 0]
                train_data0 = ori_data
                pass_data = ori_data[int(sample_num * 0.8):, :]
                indices = [i for i in range(1, pass_data.shape[1])]
                pass_data = increment(pass_data, indices)

            elif data_name == 'Fatigue':
                ori_data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/fatigue.xlsx', header=0,
                                         index_col=0)
                #train_data = ori_data.to_numpy()[1:11, :11]
                #pass_data = ori_data.to_numpy()[11:, :11]  # * 10
                #constant = ori_data.to_numpy()[1:11, 0]  # * 10
                constant = ori_data.to_numpy()[1:17, 0]  # * 10
                train_data = ori_data.to_numpy()[1:17, :11]
                constant0 = ori_data.to_numpy()[1:, 0]
                train_data0 = ori_data.to_numpy()[1:, :11]
                pass_data = ori_data.to_numpy()[17:, :11]
                indices = [i for i in range(1, pass_data.shape[1])]
                # train_data = increment(train_data, indices)
                pass_data = increment(pass_data, indices)
                # train_data = add_index(train_data)
                # print("shape of Z:", np.shape(train_data))

            elif data_name == 'Train wheel':
                ori_data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/trainwheel.xlsx', header=0)
                constant = ori_data.to_numpy()[:8, 0]  # * 10
                train_data = ori_data.to_numpy()[:8, :]
                constant0 = ori_data.to_numpy()[:11, 0]
                train_data0 = ori_data.to_numpy()[:11, :]
                pass_data = ori_data.to_numpy()[8:11, :]
                indices = [i for i in range(1, pass_data.shape[1])]
                # train_data = increment(train_data, indices)
                pass_data = increment(pass_data, indices)

            elif data_name == 'Laser':
                ori_data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/laserdata.xlsx', header=0)
                constant = ori_data.to_numpy()[:12, 0]  # * 10
                train_data = ori_data.to_numpy()[:12, :]
                constant0 = ori_data.to_numpy()[:, 0]
                train_data0 = ori_data.to_numpy()
                pass_data = ori_data.to_numpy()[12:, :]
                indices = [i for i in range(1, pass_data.shape[1])]
                # train_data = increment(train_data, indices)
                pass_data = increment(pass_data, indices)

            elif data_name == 'Interial navigition':
                ori_data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/interial navigition.xlsx',
                                         header=0, index_col=0)
                #constant = ori_data.to_numpy()[1:3, 0]  # * 10
                #train_data = ori_data.to_numpy()[1:3, :9]  # * 10
                #pass_data = ori_data.to_numpy()[3:, :9]
                constant = ori_data.to_numpy()[:3, 0]  # * 10
                train_data = ori_data.to_numpy()[:3, :9]  # * 10
                constant0 = ori_data.to_numpy()[:, 0]
                train_data0 = ori_data.to_numpy()[:, :9]
                pass_data = ori_data.to_numpy()[3:, :9]
                indices = [i for i in range(1, pass_data.shape[1])]
                pass_data = increment(pass_data, indices)

            # pass_data = scaler.fit_transform(pass_data.T).T
            # constant = scaler.fit_transform(constant.reshape(1, -1).T).T

            Z = torch.Tensor(pass_data).float().cuda()
            Z_real = torch.Tensor(train_data).float().cuda()
            #Z_real = dataset

            fig, axs = plt.subplots(2, 10, figsize=(28, 6))
            num = 10 * int(num_steps / 100)
            for i in range(1, 11):
                cur_x = z_candidate_seq[i * num].detach()
                cur_x = cur_x.cpu().detach().numpy()
                cur_x = sample_extend0(cur_x, constant) #/ 10

                axs[0, i - 1].plot(range(1, d+2), cur_x.T, 'r.--');
                axs[0, i - 1].set_title('$p(\mathbf{x}_{' + str(i * num) + '})$')
                axs[1, i - 1].plot(range(1, d+2), train_data.T, 'g+--');
                axs[1, i - 1].set_title('$real image$')

            if not os.path.exists(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch)):
                os.makedirs(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch))
            plt.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/generatefig' + str(n) + '.png')
            plt.cla()
            plt.close('all')

            numb += 1
            candidate_list.append(sample_extend0(z_candidate.cpu().detach().numpy(), constant))
            Z_list.append(Z_real.cpu().detach().numpy())

        return sample_list, candidate_list, Z_list


    # 生成DDPM和PASS_DDPM数据
    z_pass_ddpm, z_ddpm, Z = PASS_DDPM(D, dataset, data_name)
    Z0 = np.array(Z) # * 50
    Z = Z0[:, :n]
    print(Z.shape)
    z_pass_ddpm0 = np.array(z_pass_ddpm) # * 50
    z_ddpm0 = np.array(z_ddpm) # * 50
    z_ddpm = z_ddpm0[:, :n]
    print(z_ddpm.shape)
    print('DDPM ends!')
    time_elapsed_ddpm = time.time() - since_ddpm

    # SVAE-GRU
    since_svae = time.time()
    ori, svae_gru, svae_gru_pre, svae_gru_in, svae_gru_data0 = svae_gru(ori_data, D, data_name, svae_epoch, svae_path, state=svae_state)
    svae_gru_data0 = svae_gru_data0.reshape(D, -1, d + 1)
    svae_gru_data = svae_gru_data0[:, :n]
    print(svae_gru_data.shape)
    print('SVAE-GRU ends!')
    time_elapsed_svae = time.time() - since_svae

    # 生成TimeGAN和PASS_TimeGAN数据
    since_timegan = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['Fatigue','Train wheel','Laser', 'Interial navigition', 'wiener', 'gamma', 'Inverse Gaussian'],
        default=data_name,
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=2,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=t_epoch,
        type=int) #3000 #5000 #
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=t_batch,
        type=int) #32 #16 #5
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)
    parser.add_argument(
        '--num_generation',
        help='number of generated data from timegan/pass',
        default=D,
        type=int)

    args = parser.parse_args()
    # Calls main function
    ori, generated_data, pass_generated_data, metric_results, generated_list, pass_generated_list = main_timegan(ori_data, args)
    #print(generated_list)
    #print(len(generated_list))

    timegan_data = []
    timegan_pass = []
    for i in range(len(generated_list)):
        timegan_data.append(generated_list[i])
        #timegan_pass.append(pass_generated_list[i])

    timegan_data = np.array(timegan_data).reshape(D, -1, d+1)
    #timegan_pass = np.array(timegan_pass).reshape(D, -1, d+1)
    print(timegan_data.shape)
    print('TimeGAN ends!')
    time_elapsed_timegan = time.time() - since_timegan

    if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
        ylab=data_name + ' degradation value'
        xlab=Time
        x_ticks = np.arange(0, N+1, 1)
        x = np.linspace(0, N, N+1)

    elif data_name == 'Fatigue':
        ylab='Length of crack'
        xlab='Million cycle'
        x_ticks = np.arange(0, .11, 0.01)
        x = np.linspace(0, 0.1, 11)

    elif args.data_name == 'Train wheel':
        ylab='Deg(mm)'
        xlab="Dis(1e4 km)"
        x_ticks = np.arange(0, 650, 50)
        x = np.linspace(0, 600, 13)

    elif args.data_name == 'Laser':
        ylab='Electric current'
        xlab="Time"
        x_ticks = np.arange(0, 4250, 250)
        x = np.linspace(0, 4000, 17)

    elif data_name == 'Interial navigition':
        ylab='Gyroscopic drift'
        xlab='Time'
        x_ticks = np.arange(2.5, 25, 2.5)
        x = np.linspace(2.5, 22.5, 9)

    Fignum = 20
    for fignum in range(Fignum):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        od = ax.plot(x, Z[fignum].T, 'bo--', label='Original')
        plt.setp(od[1:], label="_")
        ed = ax.plot(x, expand_data[fignum].T, 'm^--', label='Segmented Sampling')
        plt.setp(ed[1:], label="_")
        plt.xticks(x_ticks)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        ax.legend(prop={'size': 12})
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_ss' + str(fignum) + '.png')
        plt.cla()
        plt.close('all')

        row = [i for i in np.arange(0, n, 1)]
        column = [i for i in np.arange(0, d + 1, 1)]
        save_csv = pd.DataFrame(columns=column, data=Z[fignum])
        save_csv.index = row
        save_csv.to_csv(
            args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_original' + str(
                fignum) + '.csv')

        save_csv = pd.DataFrame(columns=column, data=expand_data[fignum])
        save_csv.index = row
        save_csv.to_csv(
            args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_ss' + str(
                fignum) + '.csv')

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        od = ax.plot(x, Z[fignum].T, 'bo--', label='Original')
        plt.setp(od[1:], label="_")
        dd = ax.plot(x, z_ddpm[fignum].T, 'g+--', label='DDPM')
        plt.setp(dd[1:], label="_")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xticks(x_ticks)
        ax.legend(prop={'size': 12})
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_ddpm' + str(fignum) + '.png')
        plt.cla()
        plt.close('all')

        save_csv = pd.DataFrame(columns=column, data=z_ddpm[fignum])
        save_csv.index = row
        save_csv.to_csv(
            args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_ddpm' + str(
                fignum) + '.csv')

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        od = ax.plot(x, Z[fignum].T, 'bo--', label='Original')
        plt.setp(od[1:], label="_")
        td = ax.plot(x, timegan_data[fignum].T, 'c*--', label='TimeGAN')
        plt.setp(td[1:], label="_")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xticks(x_ticks)
        ax.legend(prop={'size': 12})
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_timegan' + str(fignum) + '.png')
        plt.cla()
        plt.close('all')

        save_csv = pd.DataFrame(columns=column, data=timegan_data[fignum])
        save_csv.index = row
        save_csv.to_csv(
            args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_timegan' + str(
                fignum) + '.csv')

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        od = ax.plot(x, Z[fignum].T, 'bo--', label='Original')
        plt.setp(od[1:], label="_")
        sd = ax.plot(x, svae_gru_data[fignum].T, 'ys--', label='SVAE-GRU')
        plt.setp(sd[1:], label="_")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.xticks(x_ticks)
        ax.legend(prop={'size': 12})
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_svaegru' + str(fignum) + '.png')
        plt.cla()
        plt.close('all')

        save_csv = pd.DataFrame(columns=column, data=svae_gru_data[fignum])
        save_csv.index = row
        save_csv.to_csv(
            args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_svaegru' + str(fignum) + '.csv')


    import seaborn as sns
    from scipy.stats import ks_2samp, kstest
    from sklearn.neighbors import KernelDensity
    import torch.nn.functional as F
    from scipy.stats import mstats

    Z_incre, expand_data_incre, z_ddpm_incre, timegan_data_incre, svae_gru_data_incre = [], [], [], [], []
    stat1, stat2, stat3, stat4 = [], [], [], []
    P1, P2, P3, P4 = [], [], [], []
    if data_name == 'Inverse Gaussian':
        rand = invgauss
    elif data_name == 'wiener':
        rand = norm
    elif data_name == 'gamma':
        rand = gamma

    for i in range(D):

        Z_incre.append(increment(ori_data, indices))
        expand_data_incre.append(increment(expand_data[i], indices))
        z_ddpm_incre.append(increment(z_ddpm[i], indices))
        #z_pass_ddpm_incre.append(increment(z_pass_ddpm[-1], indices).flatten())
        timegan_data_incre.append(increment(timegan_data[i], indices))
        #timegan_pass_incre.append(increment(timegan_pass[-1], indices).flatten())
        svae_gru_data_incre.append(increment(svae_gru_data[i], indices))

        if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
            s1, p1 = kstest(increment(expand_data[i], indices).flatten(), rand(alpha0).cdf)
            s2, p2 = kstest(increment(z_ddpm[i], indices).flatten(), rand(alpha0).cdf)
            s3, p3 = kstest(increment(timegan_data[i], indices).flatten(), rand(alpha0).cdf)
            s4, p4 = kstest(increment(svae_gru_data[i], indices).flatten(), rand(alpha0).cdf)
        else:
            s1, p1 = ks_2samp(increment(expand_data[i], indices).flatten(), increment(ori_data, indices).flatten())
            s2, p2 = ks_2samp(increment(z_ddpm[i], indices).flatten(), increment(ori_data, indices).flatten())
            s3, p3 = ks_2samp(increment(timegan_data[i], indices).flatten(), increment(ori_data, indices).flatten())
            s4, p4 = ks_2samp(increment(svae_gru_data[i], indices).flatten(), increment(ori_data, indices).flatten())

        lst = [stat1, stat2, stat3, stat4, P1, P2, P3, P4]
        ele = [s1, s2, s3, s4, p1, p2, p3, p4]
        for item in range(len(lst)):
            lst[item].append(ele[item])

    print('K-S Statistic of Segmented sampling increment:', np.mean(stat1))
    print('K-S Statistic of DDPM increment:', np.mean(stat2))
    print('K-S Statistic of TimeGAN increment:', np.mean(stat3))
    print('K-S Statistic of SVAE-GRU increment:', np.mean(stat4))

    print('Fraction about P in K-S of Segmented sampling increment:', count_above_threshold(P1, 0.05)/len(P1))
    print('Fraction about P in K-S of DDPM increment:', count_above_threshold(P2, 0.05)/len(P2))
    print('Fraction about P in K-S of TimeGAN increment:', count_above_threshold(P3, 0.05)/len(P3))
    print('Fraction about P in K-S of SVAE-GRU increment:', count_above_threshold(P4, 0.05)/len(P4))

    Data = [expand_data, z_ddpm, timegan_data, svae_gru_data] #z_pass_ddpm, , timegan_pass
    Name = ['Segmented sampling', 'DDPM', 'TimeGAN', 'SVAE-GRU'] #'PASS-DDPM', , 'PASS-TimeGAN'
    Incre = [expand_data_incre, z_ddpm_incre, timegan_data_incre, svae_gru_data_incre]
    Incre_name = ["Segmented sampling", "DDPM", "TimeGAN", "SVAE-GRU"]
    color = ["m", "g", "c", "y"]


    def gaussian_kernel(source, target):
        '''
        将源域数据和目标域数据转化为核矩阵，即上文中的K
        Params:
         source: (b1,n)的X分布样本数组
         target:（b2，n)的Y分布样本数组
        Return:
          kernel_val: 对应的核矩阵
        '''
        # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = np.concatenate((source, target), axis=0)
        # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
        total0 = np.expand_dims(total, axis=0)
        total0 = np.broadcast_to(total0, [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
        # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
        total1 = np.expand_dims(total, axis=1)
        total1 = np.broadcast_to(total1, [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
        # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
        # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
        L2_distance_square = np.cumsum(np.square(total0 - total1), axis=2)
        # 调整高斯核函数的sigma值
        bandwidth = np.sum(L2_distance_square) / (n_samples ** 2 - n_samples)
        # 高斯核函数的数学表达式
        kernel_val = np.exp(-L2_distance_square / bandwidth)
        # 得到最终的核矩阵
        return kernel_val


    def gaussian_multi_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
        Params:
         source: (b1,n)的X分布样本数组
         target:（b2，n)的Y分布样本数组
         kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
         kernel_num: 取不同高斯核的数量
         fix_sigma: 是否固定，如果固定，则为单核MMD
        Return:
          sum(kernel_val): 多个核矩阵之和
        '''
        # 堆叠两组样本，上面是X分布样本，下面是Y分布样本，得到（b1+b2,n）组总样本
        n_samples = int(source.shape[0]) + int(target.shape[0])
        total = np.concatenate((source, target), axis=0)
        # 对总样本变换格式为（1,b1+b2,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
        total0 = np.expand_dims(total, axis=0)
        total0 = np.broadcast_to(total0, [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
        # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按复制
        total1 = np.expand_dims(total, axis=1)
        total1 = np.broadcast_to(total1, [int(total.shape[0]), int(total.shape[0]), int(total.shape[1])])
        # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
        # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
        L2_distance_square = np.cumsum(np.square(total0 - total1), axis=2)
        # 调整高斯核函数的sigma值
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance_square) / (n_samples ** 2 - n_samples)
        # 多核MMD
        # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        #print(bandwidth_list)
        # 高斯核函数的数学表达式
        kernel_val = [np.exp(-L2_distance_square / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        # 得到最终的核矩阵
        return sum(kernel_val)  # 多核合并


    def MMD(source, target):
        '''
        计算源域数据和目标域数据的MMD距离
        Params:
         source: 源域数据，行表示样本数目，列表示样本数据维度
         target: 目标域数据 同source
        Return:
         loss: MMD loss
        '''
        batch_size = int(source.shape[0])  # 一般默认为X和Y传入的样本的总批次样本数是一致的
        kernels = gaussian_kernel(source, target)
        # 将核矩阵分成4部分
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        # 这里计算出的n_loss是每个维度上的MMD距离，一般还会做均值化处理
        n_loss = loss / float(batch_size)
        return np.mean(n_loss)


    def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        计算源域数据和目标域数据的MMD距离
        Params:
         source: (b1,n)的X分布样本数组
         target:（b2，n)的Y分布样本数组
         kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
         kernel_num: 取不同高斯核的数量
         fix_sigma: 是否固定，如果固定，则为单核MMD
        Return:
         loss: MK-MMD loss
        '''
        batch_size = int(source.shape[0])  # 一般默认为源域和目标域的batchsize相同
        kernels = gaussian_multi_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        # 将核矩阵分成4部分
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        # 这里计算出的n_loss是每个维度上的MK-MMD距离，一般还会做均值化处理
        n_loss = loss / float(batch_size)
        return np.mean(n_loss)

    df, df1 = [], []
    plt.figure(figsize=(12, 12))
    if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
        num_nodes = 1000
        if data_name == 'Inverse Gaussian':
            rand = invgauss
            Z_incre0 = rand.rvs(alpha0 * Time / N, beta0 * (Time / N) ** 2, size=num_nodes)
            x = np.linspace(rand.ppf(0.00000001, alpha0), rand.ppf(0.99, alpha0), num_nodes)
        elif data_name == 'wiener':
            rand = norm
            Z_incre0 = rand.rvs(alpha0 * Time / N, beta0 * (Time / N) ** 2, size=num_nodes)
            x = np.linspace(rand.ppf(0.00000001, alpha0), rand.ppf(0.99, alpha0), num_nodes)
        elif data_name == 'gamma':
            rand = gamma
            Z_incre0 = rand.rvs(alpha0 * Time / N, scale=1 / beta0 * (Time / N) ** 2, size=num_nodes)
            x = np.linspace(rand.ppf(0.00000001, alpha0), rand.ppf(0.99999999, alpha0), num_nodes)
        data0 = np.array(Z[0])
        pdf = rand.pdf(x, alpha0)
        plt.plot(x, pdf, 'b', linewidth=2, label=data_name + ' PDF')
        df.append(x)
        df.append(pdf)
        df1.append(x)
        df1.append(pdf)

        for k in range(len(Incre)):  # Draw Plot
            data = Incre[k]
            datad = Data[k]
            kde_values = np.zeros((len(data), num_nodes))
            kl = []
            distance = []
            Mmd = []
            for j in range(len(data)):
                kde = gaussian_kde(data[j].flatten())
                if data_name == 'Inverse Gaussian':
                    x = np.linspace(np.min(data), min(np.max(data), 0.25), num_nodes)
                elif data_name == 'wiener':
                    x = np.linspace(np.min(data), np.max(data), num_nodes)
                elif data_name == 'gamma':
                    x = np.linspace(np.min(data), np.max(data), num_nodes)
                y = kde(x)
                #kl_divergence = entropy(Z_incre0, y, base=None)
                logp_x = F.log_softmax(torch.tensor(pdf) + 1e-9, dim=-1) #Z_incre[j].flatten()[:n * d]
                p_y = F.softmax(torch.tensor(y), dim=-1) #data[j].flatten()
                kl_divergence = F.kl_div(logp_x, p_y, reduction='batchmean')

                dist = np.linalg.norm(data[j].flatten() - Z_incre[j].flatten()[:n * d])
                kl.append(kl_divergence)
                mmd = np.abs(MK_MMD(data0, datad[j])) ** 0.5
                # mmd = MK_MMD(Z_incre[j][:n, :], data[j])
                #emd_distance = wasserstein_distance(data0, datad[j])
                kl.append(np.array(kl_divergence))
                distance.append(dist)
                Mmd.append(mmd)
                #Emd.append(emd_distance)
                # y = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
                for i in range(num_nodes):
                    kde_values[j, i] = y[i]

            kl = np.mean(np.array(kl))
            distance = np.mean(np.array(distance))
            Mmd = np.mean(np.array(Mmd))
            #Emd = np.mean(np.array(Emd))
            print('KL Distance of ' + Incre_name[k] + ' increment:', kl)
            print('Euclidean Distance of ' + Incre_name[k] + ' increment:', distance)
            print('MMD Distance of ' + Incre_name[k] + ' increment:', Mmd)
            #print('EMD Distance of ' + Incre_name[k] + ' increment:', Emd)
            mean_kde_values = np.mean(kde_values, axis=0)
            sns.lineplot(x=x, y=mean_kde_values.T, color=color[k], label=Incre_name[k])
            df.append(x)
            df.append(mean_kde_values)

        plt.grid(ls='--')
        plt.legend(prop={'size': 16})
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_incre.png')
        plt.cla()
        plt.close('all')
        #####################
        plt.figure(figsize=(12, 12))
        x = np.linspace(rand.ppf(0.00000001, alpha0), rand.ppf(0.99, alpha0), num_nodes)
        Z_incre0 = rand.rvs(alpha0 * Time / N, beta0 * (Time / N) ** 2, size=num_nodes)
        pdf = rand.pdf(x, alpha0)
        plt.plot(x, pdf, 'b', linewidth=2, label=data_name + ' PDF')
        for k in range(len(Incre)):  # Draw Plot
            data = Incre[k]
            datad = Data[k]
            kde_values = np.zeros((len(data), num_nodes))
            for j in range(len(data)):
                kde = gaussian_kde(data[j].flatten())
                x = np.linspace(max(np.min(data), -0.20), min(np.max(data), 0.25), num_nodes)
                y = kde(x)
                for i in range(num_nodes):
                    kde_values[j, i] = y[i]
            mean_kde_values = np.mean(kde_values, axis=0)
            sns.lineplot(x=x, y=mean_kde_values.T, color=color[k], label=Incre_name[k])
            df1.append(x)
            df1.append(mean_kde_values)
        plt.grid(ls='--')
        plt.legend(prop={'size': 16})
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_incre_cut.png')
        plt.cla()
        plt.close('all')

        row = ['x', 'original', 'x', 'segmented', 'x', 'ddpm', 'x', 'timegan', 'x', 'svaegru']
        column = [i for i in np.arange(0, num_nodes, 1)]
        save_csv = pd.DataFrame(columns=column, data=df)
        save_csv.index = row
        save_csv.to_csv(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch)
                        + '/savefig_incre.csv')
        save_csv = pd.DataFrame(columns=column, data=df1)
        save_csv.index = row
        save_csv.to_csv(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch)
                        + '/savefig_incre_cut.csv')
    else:
        num_nodes = 100
        data0 = np.array(Z[0])

        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')  # 选择核和带宽
        kde.fit(data0)

        x = np.zeros((data0.shape[-1], num_nodes))
        for layer in range(data0.shape[-1]):
            #x[layer] = np.linspace(np.min(data0[:,layer]), np.max(data0[:,layer]), num_nodes)
            x[layer] = np.linspace(np.min(data0), np.max(data0), num_nodes)
        # np.linspace(np.min(data), np.max(data), data0.shape[-1])
        #print(x)
        grid = np.dstack(np.meshgrid([x[a] for a in range(data0.shape[-1])])).reshape(-1, data0.shape[-1])
        print(grid)
        log_density = kde.score_samples(grid)
        Z_y = np.exp(log_density)

        #print(Z_y)

        sns.lineplot(np.linspace(np.min(data0), np.max(data0), num_nodes), Z_y, color="b", label="Original")

        for k in range(len(Data)): # Draw Plot
            data = Data[k]
            data_inc = Incre[k]
            kde_values = np.zeros((len(data), num_nodes))
            kl = []
            distance = []
            Mmd = []
            for j in range(len(data)):
                kde.fit(data[j])
                x = np.zeros((data[j].shape[-1], num_nodes))
                for layer in range(data[j].shape[-1]):
                    #x[layer] = np.linspace(np.min(data[j][:,layer]), np.max(data[j][:,layer]), num_nodes)
                    x[layer] = np.linspace(np.min(data[j]), np.max(data[j]), num_nodes)
                grid = np.dstack(np.meshgrid([x[a] for a in range(data0.shape[-1])])).reshape(-1, data[j].shape[-1])
                log_density = kde.score_samples(grid)
                y = np.exp(log_density)

                #print(y)
                #kl_divergence = entropy(Z_y, y, base=None)
                logp_x = F.log_softmax(torch.tensor(Z_y) + 1e-9, dim=-1)
                p_y = F.softmax(torch.tensor(y), dim=-1)
                kl_divergence = F.kl_div(logp_x, p_y, reduction='batchmean')

                dist = np.linalg.norm(data[j] - Z[0])
                # mmd = MMD(data0, data[j])
                mmd = np.abs(MK_MMD(data0, data[j])) ** 0.5
                #emd_distance = wasserstein_distance(data0, data[j])
                kl.append(np.array(kl_divergence))
                distance.append(dist)
                Mmd.append(mmd)
                #Emd.append(emd_distance)
                #y = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
                for i in range(num_nodes):
                    kde_values[j, i] = y[i]

            kl = np.sum(np.mean(np.array(kl), axis=0))
            distance = np.mean(np.array(distance))
            #print(Mmd)
            Mmd = np.mean(np.array(Mmd))
            print('KL Distance of ' + Incre_name[k] + ' increment:', kl)
            print('Euclidean Distance of ' + Incre_name[k] + ' increment:', distance)
            print('MMD Distance of ' + Incre_name[k] + ' increment:', Mmd)

            mean_kde_values = np.mean(kde_values, axis=0)
            sns.lineplot(x=np.linspace(np.min(data), np.max(data), num_nodes), y=mean_kde_values.T, color=color[k], label=Incre_name[k])

        plt.legend(prop={'size': 20})
        plt.xlabel("Values")
        plt.ylabel("Density")
        plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_multi.png')
        plt.cla()
        plt.close('all')

    print(f'Generation of Segmented Sampling complete in {time_elapsed_ss // 60:.0f}m {time_elapsed_ss % 60:.0f}s')
    print(f'Generation of DDPM complete in {time_elapsed_ddpm // 60:.0f}m {time_elapsed_ddpm % 60:.0f}s')
    print(f'Generation of TimeGAN complete in {time_elapsed_timegan // 60:.0f}m {time_elapsed_timegan % 60:.0f}s')
    print(f'Generation of SVAE-GRU complete in {time_elapsed_svae // 60:.0f}m {time_elapsed_svae % 60:.0f}s')

    # Mean and Std
    def monte_carlo_simulation(data, num_sample):
        n = len(data)
        num_features = data[0].shape[1]

        mean_est = np.mean(np.mean(np.array(data), axis=1), axis=0)
        std_est = np.mean(np.std(np.array(data), axis=1), axis=0)

        return mean_est, std_est


    def wiener_para(data):
        m1, m2 = 0, 0
        t = 0
        indices = [i for i in range(1, data.shape[1])]
        incre = increment(data, indices)
        for i in range(data.shape[0]):
            m1 += data[i, -1]
            t += Time / data.shape[1]
            for j in range(data.shape[1]-1):
                m2 += incre[i, j]**2 / (Time/data.shape[1])
        mu = m1 / t
        sigma = (m2 - m1**2/t) / data.shape[1] / data.shape[0]
        return mu, sigma


    def alphafun(alpha, data, incre):
        m1, m2, m3 = 0, 0, 0

        for m in range(data.shape[0]):
            m2 += np.log(data[m, -1] / alpha)
            for n in range(data.shape[1]):
                m1 += np.log(incre[m, n]) / data.shape[1]
                m3 += digamma(alpha / data.shape[1]) / data.shape[1]
        a1 = m1 - m2 - m3
        return a1


    # 参数估计
    def gamma_mle(data, max_iters=100, tol=1e-6):
        # 初始化参数

        indices = [ind for ind in range(1, data.shape[1])]
        incre = np.abs(increment(data, indices))
        data = np.array(data)[:, 1:]
        data = np.abs(data)  # [data > 0]
        # print(data)
        # print(data.shape)

        if np.any(data <= 0) or np.any(incre <= 0):
            raise ValueError("数据中存在非正值。")

        alpha = 20 #np.mean(data)**2/np.var(data)
        beta = 1.0 #np.mean(data)/np.var(data)
        alpha1 = 1.5

        alpha_new = optimize.newton(alphafun, alpha, args=(data, incre), maxiter=max_iters, tol=tol)
        b1 = np.sum(data[:, -1])
        b2 = data.shape[0] * data.shape[1]
        beta_new = b1 / alpha_new / b2

        return alpha_new, beta_new


    # EM算法参数估计
    def inverse_gaussian_mle(data, max_iters=100, tol=1e-6):
        # 初始化参数
        mu = np.mean(data)
        shape = 2  # 初始形状参数，可以根据实际情况调整
        for i in range(max_iters):
            '''
            b1 = np.sum(data[:, -1])
            b2 = data.shape[0] * data.shape[1]
            mu_new = b1 / b2
    
            a1 = data.shape[0] * data.shape[1]
            a2 = np.sum(data[:, -1]) / mu**2 - 2 * data.shape[0] * data.shape[1]/mu + np.sum(1/incre)
            shape_new = a1 / a2
            '''
            # E步骤
            lambda_inv = np.mean(data - mu) / np.mean((data - mu)**(-1))
            # M步骤
            mu_new = np.mean(data)
            shape_new = np.sum((data - mu_new)**(-1)) / len(data)
            # 计算对数似然增加值，用于判断收敛
            log_likelihood = np.sum(invgauss.logpdf(data, mu=mu, scale=lambda_inv**(-0.5), loc=0))
            log_likelihood_new = np.sum(invgauss.logpdf(data, mu=mu_new, scale=(shape_new/lambda_inv)**(-0.5), loc=0))
            # 计算对数似然增加值，用于判断收敛
            # log_likelihood = np.sum(invgauss.logpdf(data, mu=mu, scale=shape, loc=0))
            # log_likelihood_new = np.sum(invgauss.logpdf(data, mu=mu_new, scale=shape_new, loc=0))
            if abs(log_likelihood_new - log_likelihood) < tol:
                break

            # 更新参数
            mu = mu_new
            shape = shape_new
        return mu, shape


    def gamma_para(data):
        '''
        mean = np.mean(np.mean(data, axis=1))
        var = np.mean(np.var(data, axis=1))
        alpha = mean**2 / var
        beta = mean / var
        '''
        alpha, beta = 0, 0
        alpha = np.mean(data) ** 2 / np.var(data)
        beta = np.mean(data) / np.var(data)

        # alpha, beta = gamma_mle(data)
        '''
        for i in range(data.shape[0]):
            # 进行参数估计
            alpha0, loc, beta0 = gamma.fit(data[i], floc=0)
            alpha += alpha0
            beta += beta0
        alpha /= data.shape[0]
        beta /= data.shape[0]
        '''
        return alpha, beta


    def inverse_gaussian_para(data):
        '''
        mean = np.mean(np.mean(data, axis=1))
        std = np.mean(np.std(data, axis=1))
        alpha = mean
        beta = std # mean**3/var
        '''
        mu, shape = 0, 0

        indices = [ind for ind in range(1, data.shape[1])]
        incre = increment(data, indices)
        # mu, shape = inverse_gaussian_mle(data, incre)

        b1 = np.sum(data[:, -1])
        b2 = data.shape[0] * data.shape[1]
        mu = b1 / b2

        a1 = data.shape[0] * data.shape[1]
        a2 = np.sum(data[:, -1]) / mu**2 - 2 * data.shape[0] * data.shape[1]/mu + np.sum(1/incre)
        shape = a1 / a2

        return mu, shape


    def general_statistics(data):
        dist = np.linalg.norm(data, ord=2)
        points, _ = detect_failure_points(data, threshold)
        rul = np.mean(points)
        return dist, rul


    if args.data_name == 'gamma':
        standard_para = gamma_para
        set = [0, 1]
    elif args.data_name == 'Inverse Gaussian':
        standard_para = inverse_gaussian_para
        set = [0, 1]
    elif args.data_name == 'wiener':
        standard_para = wiener_para
        set = [0, 1]
    else:
        standard_para = general_statistics
        set = [1]


    def monte_carlo_simulation_standard(data, num_sample):
        n = len(data)
        alpha_est = 0
        beta_est = 0

        for i in range(n):
            # print(standard_para(sample[i]))
            sample_alpha_est = standard_para(np.array(data)[i])[0]
            sample_beta_est = standard_para(np.array(data)[i])[1]
            alpha_est += sample_alpha_est
            beta_est += sample_beta_est

        alpha_est /= (num_sample)
        beta_est /= (num_sample)

        return alpha_est, beta_est


    mean_real, std_real = np.mean(np.mean(Z, axis=1), axis=0), np.mean(np.std(Z, axis=1), axis=0)

    # 真实数据的特征
    print("mean of Z:", mean_real)
    print("std of Z:", std_real)

    num_sample = 1000  # 模拟次数

    alpha_real_est, beta_real_est = monte_carlo_simulation_standard(Z, num_sample)
    print(alpha_real_est)

    # 对截断生成样本进行MCMC
    mean_ext_est, std_ext_est = monte_carlo_simulation(expand_data, num_sample)
    alpha_ext_est, beta_ext_est = monte_carlo_simulation_standard(expand_data, num_sample)
    print(alpha_ext_est)

    print("mean estimation of Segmented Sampling with MCMC1000:", mean_ext_est)
    print("std estimation of Segmented Sampling with MCMC1000:", std_ext_est)

    # 对DDPM生成样本进行MCMC
    mean_ddpm_est, std_ddpm_est = monte_carlo_simulation(z_ddpm, num_sample)
    alpha_ddpm_est, beta_ddpm_est = monte_carlo_simulation_standard(z_ddpm, num_sample)
    print(alpha_ddpm_est)

    print("mean estimation of DDPM with MCMC1000:", mean_ddpm_est)
    print("std estimation of DDPM with MCMC1000:", std_ddpm_est)


    # 对TimeGAN生成样本进行MCMC
    mean_timegan_est, std_timegan_est = monte_carlo_simulation(timegan_data, num_sample)
    alpha_timegan_est, beta_timegan_est = monte_carlo_simulation_standard(timegan_data, num_sample)
    print(alpha_timegan_est)

    print("mean estimation of TimeGAN with MCMC1000:", mean_timegan_est)
    print("std estimation of TimeGAN with MCMC1000:", std_timegan_est)

    # 对SVAE-GRU生成样本进行MCMC
    mean_svaegru_est, std_svaegru_est = monte_carlo_simulation(svae_gru_data, num_sample)
    alpha_svaegru_est, beta_svaegru_est = monte_carlo_simulation_standard(svae_gru_data, num_sample)
    print(alpha_svaegru_est)

    print("mean estimation of SVAE-GRU with MCMC1000:", mean_svaegru_est)
    print("std estimation of SVAE-GRU with MCMC1000:", std_svaegru_est)



    def empirical_distribution(data, data0, stat_func, num_sample=D):
        stat_values = np.zeros(len(data))
        stat_values2 = np.zeros(len(data))
        if stat_func == standard_para:
            for i in range(len(data)):
                stat_values[i] = stat_func(data[i])[0]
                stat_values2[i] = stat_func(data[i])[1]

            # print(stat_values)

            kde = gaussian_kde(stat_values)
            x = np.linspace(np.min(stat_values), np.max(stat_values), num_sample)

            y = kde(x)
            y = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))

            kde2 = gaussian_kde(stat_values2)
            x2 = np.linspace(np.min(stat_values2), np.max(stat_values2), num_sample)

            y2 = kde(x2)
            y2 = np.vectorize(lambda x: kde2.integrate_box_1d(-np.inf, x))

        else:
            for i in range(len(data)):
                stat_values[i] = stat_func(data[i])
            #stat_values[i] = stat_func(data[i][i % data[i].shape[0]])
            #stat_values[i] = stat_func(data[i], data0[i])

            kde = gaussian_kde(stat_values)
            x = np.linspace(np.min(stat_values), np.max(stat_values), num_sample)

            y = kde(x)
            y = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
            x2 = 0
            y2 = 0

        return x, y, x2, y2, stat_values, stat_values2



    ###Save

    df, df1 = [], []
    row = ['mean of Z', 'std of Z', 'mean estimation of Z_Expansion', 'std estimation of Z_Expansion',
           'mean estimation of Z_DDPM', 'std estimation of Z_DDPM',
           'mean estimation of Z_TimeGAN', 'std estimation of Z_TimeGAN',
           'mean estimation of Z_SVAE-GRU', 'std estimation of Z_SVAE-GRU']
    #'mean estimation of Z_PASS_DDPM', 'std estimation of Z_PASS_DDPM','mean estimation of Z_PASS_TimeGAN', 'std estimation of Z_PASS_TimeGAN','P-value of DDPM', 'P-value of PASS-DDPM', 'P-value of TimeGAN', 'P-value of PASS-TimeGAN'

    row1 = ['alpha', 'beta', 'alpha estimation of Z_Expansion', 'beta estimation of Z_Expansion',
           'alpha estimation of Z_DDPM', 'beta estimation of Z_DDPM',
           'alpha estimation of Z_TimeGAN', 'beta estimation of Z_TimeGAN',
            'alpha estimation of Z_SVAE-GRU', 'beta estimation of Z_SVAE-GRU']

    for i in [mean_real, std_real,
              mean_ext_est, std_ext_est,
              mean_ddpm_est, std_ddpm_est,
              mean_timegan_est, std_timegan_est,
              mean_svaegru_est, std_svaegru_est]:
        #mean_pass_ddpm_est, std_pass_ddpm_est,,P_ddpm, P_pass_ddpm, P_timegan, P_pass_timegan
        df.append(i)

    for i in [alpha_real_est, beta_real_est,
              alpha_ext_est, beta_ext_est,
              alpha_ddpm_est, beta_ddpm_est,
              alpha_timegan_est, beta_timegan_est,
              alpha_svaegru_est, beta_svaegru_est]:
        #, alpha_pass_ddpm_est, beta_pass_ddpm_est,P_ddpm, alpha_pass_timegan_est, beta_pass_timegan_est, P_pass_ddpm, P_timegan, P_pass_timegan
        df1.append(i)

    if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
        column = [i for i in np.arange(0, N+1, 1)]

    elif data_name == 'Fatigue':
        column = [i for i in np.arange(0, .11, 0.01)]

    elif data_name == 'Train wheel':
        column = [i for i in np.arange(0, 650, 50)]

    elif data_name == 'Laser':
        column = [i for i in np.arange(0, 4250, 250)]

    elif data_name == 'Interial navigition':
        column = [i for i in np.arange(2.5, 25, 2.5)]

    column1 = [i for i in np.arange(0, 1)]

    save_csv = pd.DataFrame(columns=column, data=df)
    save_csv.index = row
    save_csv.to_csv(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save.csv')

    save_csv1 = pd.DataFrame(columns=column1, data=df1)
    save_csv1.index = row1
    save_csv1.to_csv(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_para.csv')


    if data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
        plt.ylabel(data_name + ' process')
        plt.xlabel("Time")
        x_ticks = np.arange(0, N+1, 1)
        x = np.linspace(0, N, N+1)
    elif data_name == 'Fatigue':
        plt.ylabel('Increment length of crack')
        plt.xlabel("Million cycles")
        x_ticks = np.arange(0, .11, 0.01)
        x = np.linspace(0, 0.1, 11)
    elif args.data_name == 'Train wheel':
        plt.ylabel('Deg(mm)')
        plt.xlabel("Dis(1e4 km)")
        x_ticks = np.arange(0, 650, 50)
        x = np.linspace(0, 600, 13)
    elif args.data_name == 'Laser':
        plt.ylabel('Electric current')
        plt.xlabel("Time")
        x_ticks = np.arange(0, 4250, 250)
        x = np.linspace(0, 4000, 17)
    elif data_name == 'interial_navigition':
        plt.ylabel('Gyroscopic drift')
        plt.xlabel("Time")
        x_ticks = np.arange(2.5, 25, 2.5)
        x = np.linspace(2.5, 22.5, 9)

    label = [str for str in ['Original', 'Segmented sampling', 'DDPM', 'TimeGAN', 'SVAE-GRU']]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(int(len(df)/2)):
        ax.plot(x, df[2*i], label=label[i])

    plt.xticks(x_ticks)
    ax.legend(prop={'size': 12})
    plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_mean' + str(n) + '.png')
    plt.cla()
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(int(len(df) / 2)):
        ax.plot(x, df[2 * i+1], label=label[i])

    plt.xticks(x_ticks)
    ax.legend(prop={'size': 12})
    plt.savefig(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/savefig_std' + str(n) + '.png')
    plt.cla()
    plt.close('all')


    ### Prediction & evaluation
    Data0 = [expand_data, z_ddpm, timegan_data, svae_gru_data]
    #Data0 = [expand_data0, z_ddpm, timegan_data, svae_gru_data]
    #lengths = 4#2
    targets = 1
    D = D_pred # deplicate prediction D times

    num_bootstrap_samples = 10 # bootstrap num #100

    GRU_mae, GRU_rmse, LSTM_mae, LSTM_rmse, Transformer_mae, Transformer_rmse = [],[],[],[],[],[]
    GRU_mae_std, GRU_rmse_std, LSTM_mae_std, LSTM_rmse_std, Transformer_mae_std, Transformer_rmse_std = [], [], [], [],[],[]
    GRU_mape, GRU_smape, LSTM_mape, LSTM_smape, Transformer_mape, Transformer_smape = [], [], [], [],[],[]
    GRU_mape_std, GRU_smape_std, LSTM_mape_std, LSTM_smape_std, Transformer_mape_std, Transformer_smape_std = [], [], [], [],[],[]
    GRU_crps, LSTM_crps, Transformer_crps = [], [], []
    RUL = []
    GRU_pre, LSTM_pre, Transformer_pre = [], [], []

    for i in range(len(Data0)):
        data = Data0[i]
        gen_name = Name[i]

        (ori_pre_gru, ori_pre_lstm, ori_pre_trsf, ori_gru_mae, ori_gru_rmse, ori_lstm_mae, ori_lstm_rmse, ori_trsf_mae, ori_trsf_rmse,
         ori_gru_mape, ori_gru_smape, ori_lstm_mape, ori_lstm_smape, ori_trsf_mape, ori_trsf_smape, ori_gru_crps, ori_lstm_crps, ori_trsf_crps,
         pre_gru, pre_lstm, pre_trsf, gru_mae, gru_rmse, lstm_mae, lstm_rmse, trsf_mae, trsf_rmse,
         gru_mape, gru_smape, lstm_mape, lstm_smape, trsf_mape, trsf_smape, gru_crps, lstm_crps, trsf_crps,
         num0) = prediction(data, ori_data[:n], test_data, lengths, targets, BATCH_SIZE, D,
                            num_bootstrap_samples, data_name, gen_name, num_steps, num_epoch)

        lst = [GRU_mae, GRU_rmse, LSTM_mae, LSTM_rmse, Transformer_mae, Transformer_rmse,
               GRU_mape, GRU_smape, LSTM_mape, LSTM_smape, Transformer_mape, Transformer_smape,
               GRU_crps, LSTM_crps, Transformer_crps, GRU_pre, LSTM_pre, Transformer_pre]
        ele = [gru_mae, gru_rmse, lstm_mae, lstm_rmse, trsf_mae, trsf_rmse,
               gru_mape, gru_smape, lstm_mape, lstm_smape, trsf_mape, trsf_smape,
               gru_crps, lstm_crps, trsf_crps, pre_gru, pre_lstm, pre_trsf]
        ori_ele = [ori_gru_mae, ori_gru_rmse, ori_lstm_mae, ori_lstm_rmse, ori_trsf_mae, ori_trsf_rmse,
                   ori_gru_mape, ori_gru_smape, ori_lstm_mape, ori_lstm_smape, ori_trsf_mape, ori_trsf_smape,
                   ori_gru_crps, ori_lstm_crps, ori_trsf_crps, ori_pre_gru, ori_pre_lstm, ori_pre_trsf]

        if i == 0:
            for item in range(len(lst)):
                lst[item].append(np.array(ori_ele[item]))

        for item in range(len(lst)):
            lst[item].append(np.array(ele[item]))

    print(Z.shape)
    print(np.array(GRU_pre).shape)
    print(np.array(GRU_mae).shape)
    #print(np.array(GRU_mae))
    num1, num2, num3, num4 = 1.05, 1, 2, 0
    l = np.array(GRU_pre).shape[3]

    for n_boot in range(num_bootstrap_samples):
        for a in range(test_data.shape[0]):
            x = torch.arange(0, d+1, dtype=torch.float32)
            o = plt.plot(x[-l:], test_data[a, -l:].T, 'ko-', label='Real data')
            plt.setp(o[1:], label="_")
            pre = plt.plot(x[-l:], np.array(GRU_pre)[:, n_boot, a].squeeze().T, '*--',
                           label=[str + '+GRU' for str in ['Original']+Name])
            pre2 = plt.plot(x[-l:], np.array(LSTM_pre)[:, n_boot, a].squeeze().T, '+--',
                            label=[str + '+LSTM' for str in ['Original']+Name])
            pre3 = plt.plot(x[-l:], np.array(Transformer_pre)[:, n_boot, a].squeeze().T, '^--',
                            label=[str + '+Transformer' for str in ['Original'] + Name])

            plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
            plt.xlabel('Time Step')
            plt.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/predictionfig_gru_lstm_trsf'+str(a+1)+'.png', bbox_inches='tight')
            plt.cla()
            plt.close('all')


        #将横坐标转换为数值
        label = ['Original', 'Original+Segmented Sampling', 'Original+DDPM', 'Original+TimeGAN', 'Original+SVAE-GRU']
        xlabels = ['GRU', 'LSTM', 'Transformer']
        plt.figure(figsize=(16, 12))
        x = np.arange(3) * 2
        width = 0.2
        for i in range(len(GRU_rmse)):
            x1 = x + (i-3) * width
            a1 = [GRU_rmse[i][n_boot], LSTM_rmse[i][n_boot], Transformer_rmse[i][n_boot]]
            plt.bar(x1, a1, width=width, label=label[i])
        plt.xticks(x, labels=xlabels, size=20)
        plt.legend(prop={'size': 14}, loc='upper center')
        plt.xlabel('Prediction type', fontsize=20)
        plt.ylabel('Predictive error', fontsize=20)
        plt.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/predictionfig_rmse_all' + str(n_boot) +'.png')
        plt.cla()
        plt.close('all')

        plt.figure(figsize=(16, 12))
        for i in range(len(GRU_mae)):
            x1 = x + (i-3) * width
            a1 = [GRU_mae[i][n_boot], LSTM_mae[i][n_boot], Transformer_mae[i][n_boot]]
            plt.bar(x1, a1, width=width, label=label[i])
        plt.xticks(x, labels=xlabels, size=20)
        plt.legend(prop={'size': 14}, loc='upper center')
        plt.xlabel('Prediction type', fontsize=20)
        plt.ylabel('Predictive error', fontsize=20)
        plt.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/predictionfig_mae_all' + str(n_boot) +'.png')
        plt.cla()
        plt.close('all')

        plt.figure(figsize=(16, 12))
        for i in range(len(GRU_mape)):
            x1 = x + (i-3) * width
            a1 = [GRU_mape[i][n_boot], LSTM_mape[i][n_boot], Transformer_mape[i][n_boot]]
            plt.bar(x1, a1, width=width, label=label[i])
        plt.xticks(x, labels=xlabels)
        plt.legend(prop={'size': 14}, loc='upper center')
        plt.xlabel('Prediction type')
        plt.ylabel('Predictive error')
        plt.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/predictionfig_mape_all' + str(n_boot) +'.png')
        plt.cla()
        plt.close('all')

        plt.figure(figsize=(16, 12))
        for i in range(len(GRU_smape)):
            x1 = x + (i-3) * width
            a1 = [GRU_smape[i][n_boot], LSTM_smape[i][n_boot], Transformer_smape[i][n_boot]]
            plt.bar(x1, a1, width=width, label=label[i])
        plt.xticks(x, labels=xlabels)
        plt.legend(prop={'size': 14}, loc='upper center')
        plt.xlabel('Prediction type')
        plt.ylabel('Predictive error')
        plt.savefig(data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/predictionfig_smape_all' + str(n_boot) +'.png')
        plt.cla()
        plt.close('all')

        df2 = []
        row2 = ['MAE of GRU', 'RMSE of GRU', 'MAPE of GRU',
                'SMAPE of GRU', 'MAE of LSTM', 'RMSE of LSTM',
                'MAPE of LSTM', 'SMAPE of LSTM',
                'MAE of Transformer', 'RMSE of Transformer',
                'MAPE of Transformer', 'SMAPE of Transformer'
                ]
        #, 'CRPS of GRU', 'CRPS of LSTM'
        for i in [GRU_mae, GRU_rmse, GRU_mape, GRU_smape,
                  LSTM_mae, LSTM_rmse, LSTM_mape, LSTM_smape,
                  Transformer_mae, Transformer_rmse, Transformer_mape, Transformer_smape]:
            #, GRU_crps, LSTM_crps
            df2.append(np.array(i)[:, n_boot])

        column2 = ['Original']+Name

        save_csv2 = pd.DataFrame(columns=column2, data=df2)
        save_csv2.index = row2
        save_csv2.to_csv(args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict' + str(n_boot) +'.csv')

    row3 = [i for i in range(1, num_bootstrap_samples+1)]
    column3 = ['Original'] + Name
    save_csv3 = pd.DataFrame(columns=column3, data=np.array(GRU_mae).T)
    save_csv3.index = row3
    save_csv3.to_csv(
        args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict_GRU_mae.csv')

    save_csv3 = pd.DataFrame(columns=column3, data=np.array(GRU_rmse).T)
    save_csv3.index = row3
    save_csv3.to_csv(
        args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict_GRU_rmse.csv')

    save_csv3 = pd.DataFrame(columns=column3, data=np.array(LSTM_mae).T)
    save_csv3.index = row3
    save_csv3.to_csv(
        args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict_LSTM_mae.csv')

    save_csv3 = pd.DataFrame(columns=column3, data=np.array(LSTM_rmse).T)
    save_csv3.index = row3
    save_csv3.to_csv(
        args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict_LSTM_rmse.csv')

    save_csv3 = pd.DataFrame(columns=column3, data=np.array(Transformer_mae).T)
    save_csv3.index = row3
    save_csv3.to_csv(
        args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict_Transformer_mae.csv')

    save_csv3 = pd.DataFrame(columns=column3, data=np.array(Transformer_rmse).T)
    save_csv3.index = row3
    save_csv3.to_csv(
        args.data_name + '/generate_' + str(num_steps) + '_' + str(num_epoch) + '/save_predict_Transformer_rmse.csv')