import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from defaults import get_cfg
from torchvision.models import resnet
from torch.autograd import Variable
from scipy.stats import norm


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.ConvTranspose2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    else:
        layers.append(nn.Sigmoid())

    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers

def linear_bn_relu(input_size, output_size, slope=0.2, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Linear(input_size, output_size)
    )
    if bn:
        layers.append(nn.BatchNorm1d(output_size))
    if relu:
        layers.append(nn.LeakyReLU(slope, inplace=True))
    else:
        layers.append(nn.Sigmoid())

    layers = nn.Sequential(*layers)
    # for m in layers.modules():
    #     init_weights(m)
    return layers


class cvae(nn.Module):
    def __init__(self, cfg):
        super(cvae, self).__init__()
        self.cfg = cfg
        self.latent_size = cfg.model.latent_size
        self.input_size = cfg.data.size**2
        self.intermediate_dim = [256, 128]

        # encoder  x
        encoder = []
        self.encoder = nn.Sequential(
            linear_bn_relu(self.input_size, self.intermediate_dim[0]),
            linear_bn_relu(self.intermediate_dim[0], self.intermediate_dim[1]),
        )

        self.fc_mu = nn.Linear(self.intermediate_dim[1], self.latent_size)
        self.fc_sigma = nn.Linear(self.intermediate_dim[1], self.latent_size)

        # encoder for y
        self.fc3 = nn.Linear(10, self.intermediate_dim[1])
        self.fc4 = nn.Linear(self.intermediate_dim[1], self.latent_size)

        # decoder
        self.decoder = nn.Sequential(
            linear_bn_relu(self.latent_size, self.intermediate_dim[-1]),
            linear_bn_relu(self.intermediate_dim[-1], self.intermediate_dim[-2]),
            nn.Linear(self.intermediate_dim[-2], self.input_size),
            nn.Sigmoid()
        )

    def reparamized_trick(self, mu, log_var_2):
        epsilon = Variable(torch.randn(mu.shape)).to(mu.device)
        return mu + epsilon * torch.exp(log_var_2 / 2)

    def forward(self, x, y):
        x = x.view(x.shape[0], -1)
        # x = x.unsqueeze(1)
        y_mu = self.fc3(y)
        y_mu = self.fc4(y_mu)

        x = self.encoder(x)
        x_mu = self.fc_mu(x)
        log_var_2 = self.fc_sigma(x)
        z = self.reparamized_trick(x_mu, log_var_2)
        x = self.decoder(z)

        return x, x_mu, log_var_2, y_mu

    def encode(self, x):
        x = x.view(x.shape[0], -1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var_2 = self.fc_sigma(x)
        z = self.reparamized_trick(mu, log_var_2)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x.view(-1, 1, self.cfg.data.size, self.cfg.data.size)

    def generate(self, cfg, target, H, W):
        assert self.latent_size == 2, 'Dimension != 2, check the models latent code size!'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.eye(10).view(1, 1, 10, 10).to(device)

        x = self.fc3(x)
        mu = self.fc4(x).detach().cpu().numpy()

        digit_size = cfg.data.size
        figure = np.zeros((digit_size * H, digit_size * W))

        grid_x = norm.ppf(np.linspace(0.05, 0.95, H)) + mu[0, 0, target, 0]
        grid_y = norm.ppf(np.linspace(0.05, 0.95, W)) + mu[0, 0, target, 1]

        # grid_x = norm.ppf(np.linspace(0.05, 0.95, H))
        # grid_y = norm.ppf(np.linspace(0.05, 0.95, W))

        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]]).float().to(device)

                # z_sample = torch.tensor(z_sample).float()
                x_decoded = self.decode(z_sample.view(1, self.latent_size))
                digit = x_decoded[0, 0, :].detach().clone().cpu().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        gp = cfg.generated_path + "_" + cfg.data.data_name
        if not os.path.exists(gp):
            os.mkdir(gp)
        plt.savefig(os.path.join(gp, 'mnist_{}.png'.format(target)), format='png', dpi=400)
        # plt.show()
        plt.close()

# x = torch.randn(1,3,28,28)
# cfg = get_cfg()
# model = VAE(cfg)
# output = model.decode(torch.randn(1,32))
# print(output.shape)