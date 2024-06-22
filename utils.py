
## Necessary Packages
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]
  
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time (data):
  """Returns Maximum sequence length and each sequence length.
  
  Args:
    - data: original data
    
  Returns:
    - time: extracted time information
    - max_seq_len: maximum sequence length
  """
  time = list()
  max_seq_len = 0
  for i in range(len(data)):
    max_seq_len = max(max_seq_len, len(data[i][:,0]))
    time.append(len(data[i][:,0]))
    
  return time, max_seq_len


def rnn_cell(module_name, hidden_dim):
  """Basic RNN Cell.
    
  Args:
    - module_name: gru, lstm, or lstmLN
    
  Returns:
    - rnn_cell: RNN Cell
  """
  assert module_name in ['gru','lstm','lstmLN']
  
  # GRU
  if (module_name == 'gru'):
    rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM
  elif (module_name == 'lstm'):
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  # LSTM Layer Normalization
  elif (module_name == 'lstmLN'):
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden_dim, activation=tf.nn.tanh)
  return rnn_cell


def random_generator (batch_size, z_dim, T_mb, max_seq_len):
  """Random vector generation.
  
  Args:
    - batch_size: size of the random vector
    - z_dim: dimension of random vector
    - T_mb: time information for the random vector
    - max_seq_len: maximum sequence length
    
  Returns:
    - Z_mb: generated random vector
  """
  Z_mb = list()
  for i in range(batch_size):
    temp = np.zeros([max_seq_len, z_dim])
    temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
    temp[:T_mb[i],:] = temp_Z
    Z_mb.append(temp_Z)
  return Z_mb

def params_save(data, no, dim, seq_len, p, q):
    name_list = []
    for d in range(dim):
        temp = []
        for i in range(p[d]):
            temp.append(f"ar.L{i+1}")
        for j in range(q[d]):
            temp.append(f"ma.L{j+1}")
        name_list.append(temp)
    params_list = []
    for i in range(no):
        temp = []
        for j in range(dim):
            model = ARIMA(data[i][:,j], order = (p[j],0,q[j]))
            model_fit = model.fit()
            temp.append(dict(zip(name_list[j], model_fit.params[1:-1])))
            print(model_fit.params)
        params_list.append(temp)
        #print(temp)
    return params_list

def arima_process(data_hat, no, dim, seq_len, p, q, params_list):
    S = []
    for i in range(no):
        temp = []
        for j in range(dim):
            model = ARIMA(data_hat[i][:,j], order = (p[j],0,q[j]))
            with model.fix_params(params_list[i][j]):
                model_fit = model.fit()
            y_pred = model_fit.predict()
            temp.append(y_pred)
            #print(y_pred)
        #temp.reshape(seq_len, dim)
        temp = np.reshape(temp,(seq_len, dim))
        i = 0
        temp_data = []
        while i < seq_len:
            _x = list(temp[i])
            #print(_x)
            temp_data.append(_x)
            i += 1
        #temp = (np.reshape(temp,(seq_len, dim))).tolist()
        #temp2 = list(np.reshape(temp,(seq_len, dim)))
        #print(temp2)
        S.append(np.array(temp_data))      
    return S

def fault_simul_data(data, no, dim, freq, ampli, seq_len):
    var = np.var(data)
    eps = np.random.randn(no,seq_len,dim)*pow(var,0.1)
    simu_data = []
    for i in range(no):
        x = []
        i = 0
        for j in range(seq_len):
            x.append(i)
            i += 2*(freq/20480)*np.pi
        x = np.asarray(x)
        theta = np.random.random()*2*np.pi
        y1 = ampli*np.sin(freq*x+theta)
        y2 = ampli*np.sin(freq*x+theta+np.pi)
        y = np.vstack((y1,y2)).reshape(seq_len,-1)
        simu_data.append(y)
    simu_data = simu_data + eps
    return simu_data
     
def GP_Kron(data, no, dim, seq_len):
    D = np.reshape(data, (no,-1), order = "F")
    miu = np.average(D, axis = 0)
    cov_hat = np.cov(D, rowvar = False)
    #d_GP = np.random.multivariate_normal(miu, cov_hat, size = no)
    alpha = 0.25
    beta = 22
    d_GP = np.random.gamma(alpha, beta, size=no*dim*seq_len)
    D_GP = d_GP.reshape(no, seq_len, dim, order = "F")
    return D_GP

def batch_generator(data, time, batch_size):
  """Mini-batch generator.
  
  Args:
    - data: time-series data
    - time: time information
    - batch_size: the number of samples in each batch
    
  Returns:
    - X_mb: time-series data in each batch
    - T_mb: time information in each batch
  """
  no = len(data)
  idx = np.random.permutation(no)
  train_idx = idx[:batch_size]     
            
  X_mb = list(data[i] for i in train_idx)
  T_mb = list(time[i] for i in train_idx)
  return X_mb, T_mb