'''GRU完整模块'''

import torch
import torch.nn as nn

INPUT_SIZE = 1# The number of expected features in the input x，就是我们表示子序列中一个数的描述的特征数量，只有一个就填1，一个数字就是1
HIDDEN_SIZE = 64# The number of features in the hidden state h，隐藏状态的特征数
NUM_LAYERS = 3
targets = 1
device = ('cuda' if torch.cuda.is_available else 'cpu')
# h0 = torch.zeros([])# h0的shape与hn的shape一样为(D * num_layers, batch_size, hidden_size)
                    # 其中的D = 2 if bidirectional=True otherwise 1，num_layers为GRU的层数
                    # 如果这边不对h0进行定义，则网络中的forward中h0可以直接用None替代，默认全零。

# 定义我们的类
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,# 传入我们上面定义的参数
            hidden_size=HIDDEN_SIZE,# 传入我们上面定义的参数
            batch_first=True,# 为什么设置为True上面解释过了
            num_layers=NUM_LAYERS
        )
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32), # 加入线性层的原因是，GRU的输出，参考官网为(batch_size, seq_len, hidden_size)
            nn.LeakyReLU(),             # 这边的多层全连接，根据自己的输出自己定义就好，
            nn.Linear(32, 16),          # 我们需要将其最后打成（batch_size, output_size）比如单值预测，这个output_size就是1，
            nn.LeakyReLU(),             # 这边我们等于targets
            nn.Linear(16, targets)      # 这边输出的（batch_size, targets）且这个targets是上面一个模块已经定义好了
        )
        self.hidden_cell = torch.zeros(3, 2, HIDDEN_SIZE).to(device)

    def forward(self, input):
        #output, self.hidden_cell = self.gru(input, self.hidden_cell)# output:(batch_size, seq_len, hidden_size)，h0可以直接None
        output, h_n = self.gru(input, None)
        # print(output.shape)
        output = output[:, -1, :]# output:(batch_size, hidden_size)
        output = self.mlp(output)# 进过一个多层感知机，也就是全连接层，output:(batch_size, output_size)
        return output

