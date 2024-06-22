"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""
import os

## Necessary Packages
import numpy as np
import pickle

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from increment import increment
#from arima_fun import diff_process
from scipy.stats import geninvgauss, gamma, invgauss


def diff_process(data):
    times = 0
    while adfuller(data)[1]>0.02:
        data = np.diff(data)
        times+=1
    return data, times

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data

def simul_data_cov(no, dim, ampli, seq_len):
    miu = np.zeros(seq_len)
    cov = 0.25*np.identity(seq_len) + 0.25
    E = np.random.multivariate_normal(miu, cov, size = 2*no)
    e = E.reshape(no, seq_len*2)
    eps = e.reshape(no, -1, dim, order = "F")
    simu_data = []
    freqs = [256, 1024, 2048, 4096]
    for freq in freqs:
        for i in range(int(no/len(freqs))):
            x = []
            i = 0
            for j in range(seq_len):
                x.append(i)
                i += 2*np.pi/20480
            x = np.asarray(x)
            theta = np.random.random()*0.25*np.pi
            y1 = ampli*np.sin(freq*x+theta)
            y2 = ampli*np.sin(freq*x+theta+np.pi)
            y = np.vstack((y1,y2)).reshape(seq_len,-1)
            simu_data.append(y)
    simu_data = simu_data + eps
    return simu_data

def simul_data(no, dim, ampli, seq_len):
    eps = np.random.randn(no, seq_len, dim)*0.1
    simu_data = []
    freqs = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5]
    for freq in freqs:
        for i in range(int(no/len(freqs))):
            x = []
            i = 0
            for j in range(seq_len):
                x.append(i)
                i += 2*freq*np.pi
            x = np.asarray(x)
            theta = np.random.random()*2*np.pi
            y1 = ampli*np.sin(freq*x+theta)
            y2 = ampli*np.sin(freq*x+theta+np.pi)
            y = np.vstack((y1,y2)).reshape(seq_len,-1)
            simu_data.append(y)
    simu_data = simu_data + eps
    return simu_data

def wiener_process(T, N, mu=0, sigma=1):
    """
    Generate a sample of the Wiener process.
    T(float): Total time interval.
    N(int): Number of timr steps.
    mu(float): Mean of the Wiener process.
    sigma(float): Standard deviation of the Wiener process.

    Returns:
        numpy.ndarray: Array of shape (N+1,).
    """
    dt = T/N
    dW = np.random.normal(loc=mu * dt, scale=sigma * dt, size=N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0.0)
    return W

def gamma_process(Time, N, alpha, beta):
    """
    Generate a sample of the Gamma process.
    Time(float): Total time interval.
    N(int): Number of timr steps.
    alpha(float): shape parameter of the Gamma process.
    sigma(float): rate parameter of the Gamma process.

    Returns:
        numpy.ndarray: Array of shape (N+1,).
    """
    dt = Time/N
    gamma_values = gamma.rvs(alpha, scale=1/beta, size=N)# * dt
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
    u_values = np.random.uniform(0, 1, size=N)
    #inverse_gaussian_values = beta * (1 + (beta**2 * u_values**2 - 2 * alpha * u_values)**0.5)
    inverse_gaussian_values = invgauss.rvs(mu=alpha*dt, scale=beta*dt**2, size=N)
    inverse_gaussian_process = np.cumsum(inverse_gaussian_values)
    inverse_gaussian_process = np.insert(inverse_gaussian_process, 0, 0.0)
    return inverse_gaussian_process

def real_data_loading (ori_data, data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['Fatigue','Train wheel','Laser','Interial navigition',"wiener","gamma","Inverse Gaussian"]
  
  if data_name == 'Fatigue':
      ori_data = pd.read_excel('fatigue.xlsx', header=0, index_col=0)

      fig, ax = plt.subplots()
      x = np.linspace(0, 0.12, 13)
      ax.plot(x, np.array(ori_data).T, 'o--')
      x_ticks = np.arange(0, .13, 0.01)
      plt.xticks(x_ticks)
      plt.xlabel("Million cycles")
      plt.ylabel('Length of crack')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      ori_data = ori_data.to_numpy()[1:, :11]

      indices = [i for i in range(1, ori_data.shape[1])]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, 9, 10)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.xlabel("Million cycles")
      plt.ylabel('Increment Length of crack')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()

  elif data_name == 'Train wheel':
      ori_data = pd.read_excel('trainwheel.xlsx', header=0)

      fig, ax = plt.subplots()
      x = np.linspace(0, 600, 13)
      ax.plot(x, np.array(ori_data).T, 'o--')
      x_ticks = np.arange(0, 600, 50)
      plt.xticks(x_ticks)
      plt.ylabel('Deg(mm)')
      plt.xlabel("Dis(1e4 km)")
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      ori_data = ori_data.to_numpy()[:10, :]

      indices = [i for i in range(1, ori_data.shape[1])]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, 550, 12)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.ylabel('Deg(mm)')
      plt.xlabel("Dis(1e4 km)")
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()

  elif data_name == 'Laser':
      ori_data = pd.read_excel('laserdata.xlsx', header=0)

      fig, ax = plt.subplots()
      x = np.linspace(0, 4000, 17)
      ax.plot(x, np.array(ori_data).T, 'o--')
      x_ticks = np.arange(0, 4000, 250)
      plt.xticks(x_ticks)
      plt.ylabel('Electric current')
      plt.xlabel("Time")
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      ori_data = ori_data.to_numpy()[:, :]

      indices = [i for i in range(1, ori_data.shape[1])]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, 3750, 16)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.ylabel('Electric current')
      plt.xlabel("Time")
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()


  elif data_name == 'Interial navigition':
      ori_data = pd.read_excel('interial navigition.xlsx', header=0, index_col=0)

      fig, ax = plt.subplots()
      x = np.linspace(2.5, 22.5, 9)
      ax.plot(x, np.array(ori_data).T, 'o--')
      x_ticks = np.arange(2.5, 25, 2.5)
      plt.xticks(x_ticks)
      plt.xlabel("Time")
      plt.ylabel('Gyroscopic drift')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      ori_data = ori_data.to_numpy()[:, :9]

      indices = [i for i in range(1, ori_data.shape[1])]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, 7, 8)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Increment Gyroscopic drift')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()

  elif data_name == "wiener":
      T = 1.0
      N = ori_data.shape[1]
      sample_num = ori_data.shape[0]
      mu = 2.0
      sigma = 1.0
      '''
      ori_data = []
      for i in range(20):
          ori_data.append(wiener_process(T, N, mu, sigma))
      ori_data = np.array(ori_data)
      '''

      fig, ax = plt.subplots()
      x = np.linspace(0, N-1, N)
      ax.plot(x, np.array(ori_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Wiener process value')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      indices = [i for i in range(1, N)]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, N-2, N-1)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Increment of Wiener process value')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()

  elif data_name == "gamma":
      T = 1.0
      N = ori_data.shape[1]
      sample_num = ori_data.shape[0]
      alpha0 = 2.0
      beta0 = 1.0
      '''
      ori_data = []
      for i in range(20):
          ori_data.append(gamma_process(T, N, alpha0, beta0))
      ori_data = np.array(ori_data)
      '''

      fig, ax = plt.subplots()
      x = np.linspace(0, N-1, N)
      ax.plot(x, np.array(ori_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Gamma process value')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      indices = [i for i in range(1, N)]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, N-2, N-1)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Increment of Gamma process value')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()

  elif data_name == "Inverse Gaussian":
      T = 1.0
      N = ori_data.shape[1]
      sample_num = ori_data.shape[0]
      alpha0 = 50.0
      beta0 = 2.0
      '''
      ori_data = []
      for i in range(20):
          ori_data.append(inverse_gaussian_process(T, N, alpha0, beta0))
      ori_data = np.array(ori_data)
      '''

      fig, ax = plt.subplots()
      x = np.linspace(0, N-1, N)
      ax.plot(x, np.array(ori_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Inverse Gaussian process value')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_real.png')
      plt.close()

      indices = [i for i in range(1, N)]
      incre_data = increment(ori_data, indices)
      fig, ax = plt.subplots()
      x = np.linspace(0, N-2, N-1)
      ax.plot(x, np.array(incre_data).T, 'o--')
      plt.xlabel("Time")
      plt.ylabel('Increment of Inverse Gaussian process value')
      # Save figures
      if not os.path.exists('figures/' + data_name + '/'):
          os.makedirs('figures/' + data_name + '/')
      fig.savefig('figures/' + data_name + '/savefig_increment.png')
      plt.close()


      # Alternating sampling
      """
      temp = ori_data[5:,1]
      ori_data[:-5,1] = temp
      ori_data = ori_data[range(0,l,10),:]
      """
  # Flip the data to make chronological data 
  #ori_data = ori_data[::-1]
      #ori_data = MinMaxScaler(ori_data)
      
      #test_data = ori_data[-640000:,4:6]
      #valid_data = ori_data[640000:-640000,6:9]
      #ori_data = ori_data[:640000,4:6]
      #test_data = test_data[:,6:9]
      #ori_data = MinMaxScaler(ori_data)  
      #test_data = MinMaxScaler(test_data) 
      #valid_data = MinMaxScaler(valid_data) 
  #####UPGRADE__ORIGINAL DATA CHANGED INTO DIFFED#####
  #ori_data = np.diff(ori_data, axis=0)
  #test_data = np.diff(test_data, axis=0) 
  # Normalize the data
      #ori_data = np.vstack((ori_data, test_data))
      
  #ori_data = MinMaxScaler(ori_data)  
  #test_data = MinMaxScaler(test_data)

  # Preprocess the dataset
  #ori_data = ori_data.T

  L = ori_data.shape[0]
  if data_name == 'interial_navigition':
      L = L+1
  i = 0
  temp_data = []
  # Cut data by sequence length
  """
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
  """
  '''
  ori = ori_data[:640000,5:6]
  #plt.plot(ori[:100])
  b4 = b4[:640000]
  times = np.arange(b4.shape[0])/20480
  freqs = nf.fftfreq(times.shape[0], times[1]-times[0])
  complex_array = nf.fft(b4[:,1])
  pows = np.abs(complex_array)
  plt.plot(freqs[freqs>0], pows[freqs>0])
  a = pows[freqs>0]
  np.where(a>2000)
  freqs[np.where(pows>1200)]
  '''
  #ori_data = MinMaxScaler(ori_data)
  step = 1
  while i < L-seq_len:
      _x = ori_data[i:i+seq_len]
      temp_data.append(_x)
      i = i + step
  # Mix the datasets (to make it similar to i.i.d)
  #idx = np.random.permutation(len(temp_data))    
  #data = []
  #for i in range(len(temp_data)):
  #    data.append(temp_data[idx[i]])
  data = temp_data
  #data = data[320:]
  """D_times = []
  for i in range(10):
      t = diff_process(data[i][:,1])
      D_times.append(t[1])
  for i in range(int(np.mean(D_times))):
      data = np.diff(data, axis = 0)
  print(f"{data_name} has got difference {int(np.mean(D_times))} times")
  """
  return data