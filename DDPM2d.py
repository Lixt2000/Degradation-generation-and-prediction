#!/usr/bin/env python
# coding: utf-8
import os.path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch

from increment import sample_extend0

device = "cuda:0" if torch.cuda.is_available() else "cpu"



#确定扩散过程任意时刻的采样值

def q_x(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt):
    """可以基于x[0]得到任意时刻t的x[t]"""
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t].to(torch.device(device))
    alphas_l_m_t = one_minus_alphas_bar_sqrt[t].to(torch.device(device))
    #alphas_t = extract(alphas_bar_sqrt, t, x_0) #得到sqrt(alphas_bar[t]),x_0的作用是传入shape
    #alphas_l_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0) #得到sqrt(1-alphas_bar[t])
    return (alphas_t * x_0 + alphas_l_m_t * noise)


#编写拟合逆扩散过程高斯分布的模型

import torch
import torch.nn as nn

class MLPDiffusion(nn.Module):
    
    def __init__(self, n_steps, d, num_units=128):
        super(MLPDiffusion, self).__init__()
        
        self.linears = nn.ModuleList(
            [
                nn.Linear(d, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, d),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )
        
    def forward(self, x, t):
        
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2*idx](x.cuda())
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        return x

#编写训练的误差函数

def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]
    
    #对一个batchsize样本生成随机的时刻t，覆盖到更多不同的t
    
    #weights = torch.ones(n_steps).expend(batch_size, -1)
    #t = torch.multinomial(weights, num_samples=1, replacement=False) #[batch_size, 1]
    t = torch.randint(0, n_steps, size=(batch_size//2,))
    t = torch.cat([t, n_steps-1-t], dim=0)
    t = t.unsqueeze(-1)
    #print(t.shape)
    
    #x0的系数
    a = alphas_bar_sqrt[t].cuda()
    
    #eps的系数
    aml = one_minus_alphas_bar_sqrt[t].cuda()
    
    #生成随机噪声eps
    e = torch.randn_like(x_0).cuda()
    
    #构造模型的输入
    x = x_0 * a + e * aml
    
    #送入模型，得到t时刻的随机噪声预测值
    output = model(x.cuda(), t.squeeze(-1).cuda()).cuda()
    
    #与真实噪声一起计算误差，求平均值
    #return (e - output).square().mean()
    return (e - output).pow(2).mean()


#编写逆扩散采样函数

def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]、...、x[0]"""
    
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample_loop_V(cur_x, model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """给定一个噪声，从x[T]恢复x[T-1]、x[T-2]、...、x[0]"""

    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    t = torch.tensor([t]).cuda()
    
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    with torch.no_grad(): #在验证和测试阶段不需要计算梯度反向传播
        eps_theta = model(x, t).cuda()
    
    mean = (1 / (1-betas[t]).sqrt()) * (x.cuda() - (coeff * eps_theta))
    
    z = torch.randn_like(x).cuda()
    sigma_t = betas[t].sqrt()
    
    sample = mean + sigma_t * z
    
    return (sample)


#开始训练模型，并打印loss及中间的重构效果

seed = 1234


class EMA():
    """构建一个参数平滑器"""
    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}
        
    def register(self, name, val):
        self.shadow[name] = val.clone()
        
    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 -self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        
print('Training model...')

'''
ema = EMA(0.5)
for name, param in model.named_parameters():
    if param.requires_grad:
        ema.register(name, param.data)
'''


def ddpm1d(dataset, num_steps, batch_size, num_epoch, constant, d, data_name):

    # 对于步骤数，可由beta、分布的均值和标准差来共同确定

    # 制定每一步的beta
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    betas = betas.cuda()

    # 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量值
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0).cuda()
    alphas_prod_p = torch.cat([torch.tensor([1]).float().cuda(), alphas_prod[:-1]], 0)  # p表示previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod).cuda()
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod).cuda()
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).cuda()

    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == one_minus_alphas_bar_sqrt.shape
    print("all the same shape:", betas.shape)

    #载入数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    plt.rc('text', color='blue')

    model = MLPDiffusion(num_steps,d)  # 输出维度是1，输入是x和step
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            # for name, param in model.named_parameters():
            # if param.requires_grad:
            # param.data = ema(name, param.data)

        # print loss
        if (t % 1000 == 0):
            print(loss)

            x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)  # 共有num_steps个元素
            dataset = dataset.cpu()
            dataset0 = sample_extend0(dataset.numpy(), constant)

            fig, axs = plt.subplots(2, 10, figsize=(28, 6))
            num = 10 * int(num_steps / 100)
            for i in range(1, 11):
                cur_x = x_seq[i * num].detach()
                cur_x = cur_x.cpu().detach().numpy()
                cur_x0 = sample_extend0(cur_x, constant)
                #cur_x = np.sort(cur_x, axis=0)
                #print(cur_x)
                axs[0, i - 1].plot(range(1, d+2), cur_x0.T, 'r.--');
                #axs[0, i - 1].scatter(range(1, 11), cur_x[:, 9], color='red', edgecolor='white');
                #axs[0, i - 1].set_axis_off();
                axs[0, i - 1].set_title('$p(\mathbf{x}_{' + str(i * num) + '})$')
                axs[1, i - 1].plot(range(1, d+2), dataset0.T, 'g+--');
                #axs[1, i - 1].scatter(range(1, 11), dataset[:, 9], color='red', edgecolor='white');
                #axs[1, i - 1].set_axis_off();
                axs[1, i - 1].set_title('$real image$')

            if not os.path.exists(data_name + '/results_' + str(num_steps) + '_' + str(num_epoch) + '_' + str(dataset.shape[0])):
                os.makedirs(data_name + '/results_' + str(num_steps) + '_' + str(num_epoch) + '_' + str(dataset.shape[0]))
            plt.savefig(data_name + '/results_' + str(num_steps) + '_' + str(num_epoch) + '_' + str(dataset.shape[0]) + '/savefig_' + str(t) + '.png')
            plt.cla()
            plt.close('all')

    return model, betas, one_minus_alphas_bar_sqrt

