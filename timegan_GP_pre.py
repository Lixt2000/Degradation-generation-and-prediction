"""
timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""
import pandas as pd
# Necessary Packages
import tensorflow.compat.v1 as tf
import torch
import scipy.stats.qmc as sq
from scipy.optimize import linear_sum_assignment

tf.get_logger().setLevel('ERROR')
#tf.disable_v2_behavior()
import tf_slim as slim
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator, GP_Kron, fault_simul_data
#from arima_fun import BIC_total
import pickle
from data_loading import wiener_process, gamma_process, inverse_gaussian_process


def timegan (ori_data, parameters):
  """TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  """
  # Initialization on the Graph
  tf.reset_default_graph()

  # Network Parameters
  hidden_dim = parameters['hidden_dim']
  num_layers = parameters['num_layer']
  iterations = parameters['iterations']
  batch_size = parameters['batch_size']
  module_name = parameters['module']
  data_name = parameters['data_name']
  D = parameters['num_generation']
  gamma = 1

  if data_name == 'gamma':
    standard_process = gamma_process
  elif data_name == 'Inverse Gaussian':
    standard_process = inverse_gaussian_process
  elif data_name == 'wiener':
    standard_process = wiener_process

  # Basic Parameters
  #test_data = ori_data[30000:40000]+ori_data[-10000:]
  #test_data = ori_data[2000:6000]+ori_data[-4000:]

  #test_data = ori_data[14]+ori_data[15:]

  #ori_data = ori_data[:5000]

  #test_data = ori_data[70000:90000]+ori_data[-20000:]

  ori_data0 = ori_data

  no, seq_len, dim = np.asarray(ori_data).shape
  print(np.asarray(ori_data).shape)
  z_dim = dim

  if data_name == 'Fatigue':
    #test_data = ori_data[0] + ori_data[11:]
    #test_data = ori_data[10:]
    #ori_data = ori_data[:10]
    #test_data = ori_data[16:]
    #ori_data = ori_data[:16]
    test_data = ori_data[int(no * 0.8):]
    ori_data = ori_data[:int((no + seq_len) * 0.8)]
  elif data_name == 'Train wheel':
    test_data = ori_data[int(no * 0.8):]
    ori_data = ori_data[:int((no + seq_len) * 0.8)]
  elif data_name == 'Laser':
    test_data = ori_data[int(no * 0.8):]
    ori_data = ori_data[:int((no + seq_len) * 0.8)]
  elif data_name == 'interial_navigition':
    #test_data = ori_data[3:]
    #ori_data = ori_data[1:3]
    test_data = ori_data[3:]
    ori_data = ori_data[:3]
  elif data_name in ['wiener', 'gamma', 'Inverse Gaussian']:
    #test_data = ori_data[10:]
    #ori_data = ori_data[:10]
    test_data = ori_data[int(no*0.8):]
    ori_data = ori_data[:int((no+seq_len)*0.8)]

  #test_data = ori_data[1340:]
  #ori_data = ori_data[60:1340]
  #print(np.asarray(ori_data).shape)
  no, seq_len, dim = np.asarray(ori_data).shape
  z_dim = dim
  no_test = np.asarray(test_data).shape[0]
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  test_time, max_seq_len_test = extract_time(test_data)
  def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    #min_val = np.min(data, axis = 0)
    data = data - min_val
      
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    #max_val = np.max(data, axis = 0)
    norm_data = data / (max_val + 1e-7)
      
    return norm_data, min_val, max_val
  
  # Normalization
  ori_data, min_val, max_val = MinMaxScaler(ori_data)
  ###OIL###
  #test_data, min_val, max_val = MinMaxScaler(test_data)
  ###IMS###
  test_data = (test_data - min_val)/(max_val+1e-10)            
  ## Build a RNN networks          
  #idx = np.random.permutation(no)    
  #data = []
  #for i in range(no):
  #    data.append(ori_data[idx[i]])
  #ori_data = data
    
  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  
  # 
  def embedder (X, T):
    """Embedding network between original feature space to latent space.
    
    Args:
      - X: input time-series features
      - T: input time information
      
    Returns:
      - H: embeddings
    """
    with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
      H = slim.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return H
      
  def recovery (H, T):   
    """Recovery network from latent space to original space.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - X_tilde: recovered data
    """     
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
      X_tilde = slim.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid) 
    return X_tilde
    
  def generator (Z, T):  
    """Generator function: Generate time-series data in latent space.
    
    Args:
      - Z: random variables
      - T: input time information
      
    Returns:
      - E: generated embedding
    """        
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
      #E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
      E = slim.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)
    return E
      
  def supervisor (H, T): 
    """Generate next sequence using the previous sequence.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """          
    with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers-1)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
      S = slim.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return S
          
  def discriminator (H, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
      d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      #Y_hat_0 = slim.fully_connected(d_outputs, 1, activation_fn=None) 
      Y_hat_0 = slim.fully_connected(d_outputs, 1, activation_fn=tf.nn.sigmoid) 
      #Y_hat_0 = tf.reshape(Y_hat_0, (-1,128))
      Y_hat = slim.fully_connected(Y_hat_0[:,:,0], 1, activation_fn=None)
      #Y_hat = slim.fully_connected(tf.flaten(Y_hat_0), 1, activation_fn=None)
    return Y_hat   
    
  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)
    
  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
    
  # Synthetic data
  X_hat = recovery(H_hat, T)
    
  # Discriminator
  Y_fake = discriminator(H_hat, T)
  Y_real = discriminator(H, T)     
  Y_fake_e = discriminator(E_hat, T)
    
  # Variables        
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator loss
  D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
  # 2. Supervised loss
  G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
    
  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2
    
  # 4. Summation
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
  # Embedder network loss
  E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10*tf.sqrt(E_loss_T0)
  E_loss = E_loss0 + 0.1*G_loss_S
    
  # optimizer
  E0_solver = tf.train.AdamOptimizer().minimize(E_loss0, var_list = e_vars + r_vars)
  E_solver = tf.train.AdamOptimizer().minimize(E_loss, var_list = e_vars + r_vars)
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars + s_vars)      
  GS_solver = tf.train.AdamOptimizer().minimize(G_loss_S, var_list = g_vars + s_vars)   
        
  ## TimeGAN training   
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  m_saver = tf.train.Saver()
   
  # Generating GP data
  #h = sess.run(H, feed_dict = {X: ori_data, T: ori_time})
  GP_data = GP_Kron(ori_data, no, dim, seq_len)
  
  # Generating fault data
  #fault_data = fault_simul_data(ori_data, no, dim, 4000, 1, seq_len)
  
  # 1. Embedding network training
  print('Start Embedding Network Training')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
    # Train embedder        
    _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})        
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) ) 
      
  print('Finish Embedding Network Training')
    
  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')
    
  for itt in range(iterations):
    # Set mini-batch
    X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)    
    # Random vector generation   
    #Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
    #Z_mb, _ = batch_generator(GP_data, ori_time, batch_size)
    # Train generator       
    #_, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
    _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={X: X_mb, T: T_mb})       
    # Checkpoint
    if itt % 1000 == 0:
      print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )
      
  print('Finish Training with Supervised Loss Only')
    
  # 3. Joint Training
  print('Start Joint Training')
  
  for itt in range(iterations):
    # Generator training (twice more than discriminator training)
    for kk in range(5):
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)               
        # Random vector generation
        #Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb, _ = batch_generator(GP_data, ori_time, batch_size)
        # Train generator
        _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
         # Train embedder        
        _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})   
        #if itt % 10 == 0:
        #    m_saver.save(sess, "./GP_10000_2/", global_step=itt).encode("utf-16", errors = "ignore").decode("utf-16", errors = "ignore")
    for d1 in range(2):
        # Discriminator training        
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
        # Random vector generation
        #Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb, _ = batch_generator(GP_data, ori_time, batch_size)
        # Check discriminator loss before updating
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        #d = sess.run(Y_real, feed_dict={X: X_mb, T: T_mb})
        #print(d)
        # Train discriminator (only when the discriminator does not work well)
        if (check_d_loss > 0.15):        
          _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    """
    for d2 in range(2):
        # Discriminator training        
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)           
        # Random vector generation
        #Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        Z_mb, _ = batch_generator(fault_data, ori_time, batch_size)
        # Check discriminator loss before updating
        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        #d = sess.run(Y_real, feed_dict={X: X_mb, T: T_mb})
        #print(d)
        # Train discriminator (only when the discriminator does not work well)
        #if (check_d_loss > 0.15):
        if (check_d_loss > 0.15):        
          _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
    """      
    # Print multiple checkpoints
    if itt % 1000 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + 
            ', d_loss: ' + str(np.round(step_d_loss,4)) + 
            ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
            ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
            ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
            ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
    if (itt+1) % 5000 == 0:
        m_saver.save(sess, "./GP_simu_better_3/", global_step=itt)
  print('Finish Joint Training')
  
  def save_variable(v, filename):
      f = open(filename,"wb")
      pickle.dump(v,f)
      f.close()
      return filename
  
  h = sess.run(H, feed_dict={X: ori_data, T: ori_time})
  h_test = sess.run(H, feed_dict={X: test_data, T: test_time})
  d = sess.run(Y_real, feed_dict={H: h, T: ori_time})
  d_test = sess.run(Y_real, feed_dict={X: test_data, T: test_time})
  recov = sess.run(X_hat, feed_dict={Z: test_data, T: test_time})
  esti = sess.run(H_hat_supervise, feed_dict={X: ori_data, T: ori_time})
  esti_test = sess.run(H_hat_supervise, feed_dict={X: test_data, T: test_time})
  #e = np.array(h[:,range(0,seq_len,10),:])-np.array(esti[:,range(0,seq_len,10),:])
  #e_test = np.array(h_test[:,range(0,seq_len,10),:])-np.array(esti_test[:,range(0,seq_len,10),:])
  e = np.array(h[:,-1,:])-np.array(esti[:,-1,:])
  e_test = np.array(h_test[:,-1,:])-np.array(esti_test[:,-1,:])
  #save_variable(test_data, "test.txt")
  #save_variable(recov, "recov.txt")
  save_variable(d_test, "E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/GP_simu_better_3/d_test.txt")
  #save_variable(esti, "esti.txt")
  save_variable(d, "E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/GP_simu_better_3/Disc.txt")
  save_variable(e_test, "E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/GP_simu_better_3/e_test.txt")
  save_variable(e, "E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/GP_simu_better_3/e.txt")#保存残差，计算残差的分布
  #save_variable(a, "a.txt")
  
  ## Synthetic data generation
  Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
  #Z_total = random_generator(no, z_dim, ori_time, max_seq_len)
  #E_hat_2 = sess.run(E_hat, feed_dict = {Z: Z_mb, T: ori_time})
  '''
  h_sampler = sq.Halton(z_dim, scramble=True)
  h = h_sampler.random(no)
  class Monge:

    def swap_rows(self, matrix, i, j):
      matrix[[i, j], :] = matrix[[j, i], :]

    def random_swap(self, matrix, i, j):
      n, d = matrix.shape
      for k in range(n):
        if k != j & k != i:
          k1 = np.random.randint(n)
          if k1 != j & k1 != i:
            self.swap_rows(matrix, k, k1)

    def bestPermutation(self, Z, h, fun):
      cost_matrix = []
      for i in range(no):
        value_list = []
        for j in range(no):
          T_Z, _, _ = fun(Z)
          #print(T_Z[[1, 2]])

          self.swap_rows(T_Z, i, j)

          op = np.linalg.norm(T_Z[i, :] - h[j, :]) ** 2 / no
          value_list.append(op)
        cost_matrix.append(value_list)
      cost_mat_array = np.array(cost_matrix)
      row_ind, col_ind = linear_sum_assignment(cost_mat_array)
      return row_ind, col_ind

  monge = Monge()

  def I(x):
    return x, 1, 1


  def saveRank(U, Z, f1, f2):  # 保序映射
    U_0 = U
    _, pi1 = monge.bestPermutation(Z, h, f1)
    _, pi2 = monge.bestPermutation(U, h, f2)

    for i in range(no):
      for (index, item) in enumerate(pi1):
        if item == i:
          j = pi2[index]
      U[j, :] = U_0[i, :]
    return U

  def OPM(U, e, t):  # 选取合适的W，使其保留U的序到V
    # 先选取加性扰动
    U_e = (U + t * e) / np.sqrt(1 + t ** 2)
    #U_e, _, _ = MinMaxScaler(U_e.numpy())

    if data_name in ['fatigue', 'interial_navigition']:
      alpha = 0.5
      beta = 22
      r = np.random.gamma(alpha, beta, size=no*dim*seq_len)
    elif data_name == 'gamma':
      alpha = 2.0
      beta = 1.0
      r = np.random.gamma(alpha, beta, size=no * dim * seq_len)
    else:
      alpha = 2.0
      beta = 1.0
      r = np.random.gamma(alpha, beta, size=no * dim * seq_len)

    r = torch.tensor(r.reshape(no, seq_len, dim, order="F")).numpy()
    #r = torch.randn(U.shape).numpy()

    _, pi3 = monge.bestPermutation(U_e, r, I)

    for i in range(no):
      j = pi3[i]
      r[j, :] = U_e[i, :]

    return r
  
  def pass_GP_Kron(ori_data, dim):
    #data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/fatigue.xlsx', header=0,index_col=0)
    #data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/interial navigition.xlsx', header=0,index_col=0)

    if data_name in ['wiener', 'gamma', 'inverse_gaussian']:
      T = 1.0
      N = 20
      mu = 2.0
      sigma = 1.0
      sample_num = 20
      ori_data = []
      for i in range(sample_num):
        ori_data.append(standard_process(T, N, mu, sigma))
      Z = np.array(ori_data)[int(sample_num / 2):, :]

      if data_name == 'gamma':
        alpha = 2.0
        beta = 1.0
        U = np.random.gamma(alpha, beta, size=no * dim * seq_len)
        e = np.random.gamma(alpha, beta, size=no * dim * seq_len)
      elif data_name == 'wiener':
        alpha = 0.5
        beta = 22
        U = np.random.gamma(alpha, beta, size=no * dim * seq_len)
      else:
        alpha = 2.0
        beta = 1.0
        U = np.random.gamma(alpha, beta, size=no * dim * seq_len)
        U = np.random.randn(no * dim * seq_len)
      


    elif data_name == 'fatigue':
      ori_data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/fatigue.xlsx', header=0,
                               index_col=0)
      Z = np.array(ori_data)[11:, :dim]
      alpha = 0.5
      beta = 22
      U = np.random.gamma(alpha, beta, size=no * dim * seq_len)
      e = np.random.gamma(alpha, beta, size=no * dim * seq_len)

    elif data_name == 'interial_navigition':
      ori_data = pd.read_excel('E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/interial navigition.xlsx',
                               header=0, index_col=0)
      Z = np.array(ori_data)[2:, :dim]
      alpha = 0.5
      beta = 22
      U = np.random.gamma(alpha, beta, size=no * dim * seq_len)
      e = np.random.gamma(alpha, beta, size=no * dim * seq_len)

    # 初始分布抽样
    # U = np.random.gamma(alpha, beta, size=no*dim*seq_len)
    U = torch.tensor(U.reshape(no, seq_len, dim, order="F"))
    #Z = data.to_numpy()[11:, :dim]
    #Z = data.to_numpy()[1:, :dim]
    Z, _, _ = MinMaxScaler(Z)


    # r
    U_r = saveRank(U, Z, I, I)  # 得到U_r(i)

    e = torch.tensor(e.reshape(no, seq_len, dim, order="F"))
    tao = 1e-3

    V = OPM(U_r, e, tao)

    return V
  '''

  generated_list = [[] for _ in range(no * D)]
  generated_list_pass = [[] for _ in range(no * D)]

  for j in range(D):

    #generate pass data
    generated_data_pass = []
    '''
    GP_data_pass = pass_GP_Kron(ori_data, dim)
    generated_data_pass_curr = sess.run(X_hat, feed_dict={Z: GP_data_pass, X: ori_data, T: ori_time})

    for i in range(no):
      temp_pass = generated_data_pass_curr[i, :ori_time[i], :]
      generated_data_pass.append(temp_pass)

    # Renormalization
    generated_data_pass = generated_data_pass * max_val
    generated_data_pass = generated_data_pass + min_val
    save_variable(generated_data_pass,
                  "E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/GP_simu_better_3/gene_data_pass.txt")
    '''

    generated_data = []
    GP_data = GP_Kron(ori_data, no, dim, seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: GP_data, X: ori_data, T: ori_time})
    #print(generated_data_curr)
    #print(ori_time)
    for i in range(no):
      temp = generated_data_curr[i, :ori_time[i], :]
      generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val
    save_variable(generated_data,
                  "E:/LXT/PASS-GitHub/degradation/GAN&timeGAN/GAN&timeGAN/GP_simu_better_3/gene_data.txt")

    generated_list[j * no: (j+1) * no] = generated_data.tolist()
    # generated_list_pass[j * no: (j + 1) * no] = generated_data_pass.tolist()

  #print(generated_data)
  return generated_data, generated_list, generated_data_pass, generated_list_pass
