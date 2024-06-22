

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import warnings

import pandas as pd
import torch
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 1. TimeGAN model
from timegan_GP_pre import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
# 4. Plot
# 5. Prediction
from gru_prediction import gru_pre

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(2023)

def main_timegan (ori_data, args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  if args.data_name in ['Fatigue','Train wheel','Laser', 'Interial navigition', 'wiener', 'gamma', 'Inverse Gaussian']:
    size = ori_data.shape[-1]
    ori_data = real_data_loading(ori_data, args.data_name, args.seq_len)
    #print(ori_data)
    #print(len(ori_data))

  print(args.data_name + ' dataset is ready.')

    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
  parameters['data_name'] = args.data_name
  parameters['num_generation'] = args.num_generation

  generated_data, generated_list, PASS_generated_data, PASS_generated_list = timegan(ori_data, parameters)

  print('Finish Synthetic Data Generation')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  #visualization(ori_data, generated_data, 'pca')
  #visualization(ori_data, generated_data, 'tsne')
  
  ## Print discriminative and predictive scores
  print(metric_results)

  # 4. Plot generated data and real data
  original_data = []
  generate_data = []
  pass_generate_data = []
  generate_list = []
  pass_generate_list = []
  fig, ax = plt.subplots()

  if args.data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
      plt.ylabel(args.data_name + ' degradation increment value')
      plt.xlabel('Time')
      x_ticks = np.arange(0, size-1, 1)
      x = np.linspace(0, size-2, size-1)

  elif args.data_name == 'Fatigue':
      plt.ylabel('Increment Length of crack')
      plt.xlabel('Million cycle')
      x_ticks = np.arange(0, .1, 0.01)
      x = np.linspace(0, 0.09, 10)

  elif args.data_name == 'Train wheel':
      plt.ylabel('Deg(mm)')
      plt.xlabel("Dis(1e4 km)")
      x_ticks = np.arange(0, 600, 50)
      x = np.linspace(0, 550, 12)

  elif args.data_name == 'Laser':
      plt.ylabel('Electric current')
      plt.xlabel("Time")
      x_ticks = np.arange(0, 4000, 250)
      x = np.linspace(0, 3750, 16)

  elif args.data_name == 'Interial navigition':
      plt.ylabel('Gyroscopic drift')
      plt.xlabel('Time')
      x_ticks = np.arange(2.5, 22.5, 2.5)
      x = np.linspace(2.5, 20, 8)

  for i in range(len(generated_data)):
      #for j in range(len(ori_data[i])):
    original_data.append(ori_data[i][0])
  od = ax.plot(x, np.array(original_data).T[:-1], 'bo--', label='Original data')
  plt.setp(od[1:], label="_")
  '''
  for i in range(len(PASS_generated_list)):
      #for j in range(len(ori_data[i])):
    pass_generate_list.append(PASS_generated_list[i][0])
  for i in range(len(PASS_generated_data)):
    pass_generate_data.append(PASS_generated_data[i][0])
  pgd = ax.plot(x, np.array(pass_generate_data).T[:-1], 'r.--', label='PASS-Generated data')
  #[:, :-1]
  plt.setp(pgd[1:], label="_")
  '''
  for i in range(len(generated_list)):
      #for j in range(len(generated_data[i])):
    generate_list.append(generated_list[i][0])
  for i in range(len(generated_data)):
    generate_data.append(generated_data[i][0])
  gd = ax.plot(x, np.array(generate_data).T[:-1], 'g+--', label='Generated data')
  #[:, :-1]
  plt.setp(gd[1:], label="_")

  plt.xticks(x_ticks)
  ax.legend()

  # Save figures
  if not os.path.exists('figures/' + args.data_name + '/'):
      os.makedirs('figures/' + args.data_name + '/')
  fig.savefig('figures/' + args.data_name + '/savefig_generation-pass.png')
  plt.close()

  return ori_data, generate_data, pass_generate_data, metric_results, generate_list, pass_generate_list

