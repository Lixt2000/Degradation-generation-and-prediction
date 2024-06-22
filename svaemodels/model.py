import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, rnn_size, hidden_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(rnn_size, hidden_size)
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, latent_size, rnn_size, ):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, rnn_size)
        nn.init.xavier_normal(self.linear1.weight)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(rnn_size, input_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class VaeGRUBlock(nn.Module):
    def __init__(self, input_size, rnn_size, latent_size, hidden_size):
        super(VaeGRUBlock, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.latent_size = latent_size

        self.encoder = Encoder(rnn_size, hidden_size)
        self.decoder = Decoder(input_size, latent_size, hidden_size)

        self.gru = nn.GRU(input_size, rnn_size, batch_first=True, num_layers=1)

        self.linear1 = nn.Linear(hidden_size, 2 * latent_size)
        nn.init.xavier_normal(self.linear1.weight)

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :, :self.latent_size]
        log_sigma = temp_out[:, :, self.latent_size:]

        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        std_z = std_z.to(DEVICE)

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x):
        in_shape = x.shape
        rnn_out, _ = self.gru(x)

        enc_out = self.encoder(rnn_out)
        sampled_z = self.sample_latent(enc_out)

        dec_out = self.decoder(sampled_z)
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)

        return dec_out, self.z_mean, self.z_log_sigma


class SVaeGRU(nn.Module):

    def __init__(self, input_size, rnn_size, latent_size, hidden_size, stack_layer=2):
        super(SVaeGRU, self).__init__()
        vaegru_block = [VaeGRUBlock(input_size, rnn_size, latent_size, hidden_size) for _ in range(stack_layer)]
        self.svae_gru = nn.ModuleList(vaegru_block)
        # self.linear_out = nn.Linear(hidden_size, input_size)
        # nn.init.xavier_normal(self.linear_out.weight)
        self.stack_layer = stack_layer

    def forward(self, x):
        cur_kld = 0
        for vae_gru_layer in self.svae_gru:
            x, cur_z_mean, cur_log_sigma = vae_gru_layer(x)
            kld = torch.mean(torch.sum(0.5 * (-cur_log_sigma + torch.exp(cur_log_sigma) + cur_z_mean ** 2 - 1), -1))
            cur_kld += kld
        return x, cur_kld





