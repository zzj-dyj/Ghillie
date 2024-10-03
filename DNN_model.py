import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import kornia
from skimage.feature import structure_tensor
from skimage.feature import structure_tensor_eigenvalues
from torch.autograd import Variable
from pytorch_msssim import SSIM, MS_SSIM
from LMN_model import Network

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


def init_weights(modules):
    pass

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel // reduction, 1, 1, 0)
        self.c2 = BasicBlockSig(channel // reduction, channel , 1, 1, 0)

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=stride, padding=(self.kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        return self.ca(self.act(self.backbone(x) + self.shortcut(x)))

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        # kernel_v = [[0, -1, 0],
        #             [0, 0, 0],
        #             [0, 1, 0]]
        # kernel_h = [[0, 0, 0],
        #             [-1, 0, 1],
        #             [0, 0, 0]]
        kernel_v = [[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]]
        kernel_h = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0, x0_v, x0_h

class Coarse_DeNoise_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super(Coarse_DeNoise_Encoder, self).__init__()

        number_f = 64
        self.in_conv = nn.Conv2d(in_channels, number_f, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.e_conv2 = ResidualBlock(number_f, number_f)
        self.e_conv3 = ResidualBlock(number_f, number_f)
        self.e_conv4 = ResidualBlock(number_f, number_f)
        self.e_conv5 = ResidualBlock(number_f, number_f)
        self.e_conv6 = ResidualBlock(number_f, number_f)
        self.e_conv7 = ResidualBlock(number_f, number_f)

        self.out_conv = nn.Conv2d(number_f, out_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.lrelu(self.in_conv(x))
        x2 = self.e_conv2(x1)
        x3 = self.e_conv3(x2)
        x4 = self.e_conv4(x3)
        x5 = self.e_conv5(x4)
        x6 = self.e_conv6(x5)
        x7 = self.e_conv7(x6)
        x_r = self.out_conv(x7)

        return x_r

class DeNoise_Encoder(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super(DeNoise_Encoder, self).__init__()

        number_f = 32
        self.in_conv = nn.Conv2d(in_channels, number_f, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.e_conv2 = ResidualBlock(number_f+1, number_f*2)
        self.e_conv3 = ResidualBlock(number_f*2, number_f*4)
        self.e_conv4 = ResidualBlock(number_f*4, number_f*4)

        self.e_conv5 = ResidualBlock(number_f * 8, number_f * 2)
        self.e_conv6 = ResidualBlock(number_f * 4, number_f)
        self.e_conv7 = ResidualBlock(number_f * 2 + 1, number_f)

        self.out_conv = nn.Conv2d(number_f, out_channels, 3, 1, 1, bias=True)

    def forward(self, x, grad):
        x1 = self.lrelu(self.in_conv(x))
        x1 = torch.cat([x1, grad], dim=1)

        x2 = self.e_conv2(x1)
        x3 = self.e_conv3(x2)
        x4 = self.e_conv4(x3)

        x5 = self.e_conv5(torch.cat([x3, x4], 1))
        x6 = self.e_conv6(torch.cat([x2, x5], 1))
        x7 = self.e_conv7(torch.cat([x1, x6], 1))

        x_r = self.out_conv(x7)

        return x_r

class SSIMLoss(nn.Module):
    def __init__(self, channels):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(data_range=1., size_average=True, channel=channels)

    def forward(self, output, target):
        ssim_loss = 1 - self.ssim(output, target)
        return ssim_loss

class DNN_Network(nn.Module):

    def __init__(self):
        super(DNN_Network, self).__init__()
        self.CDN = Coarse_DeNoise_Encoder(in_channels=1, out_channels=1)
        self.DNN = DeNoise_Encoder(in_channels=1, out_channels=1)
        self.get_grad = Get_gradient()

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.ssim_loss = SSIMLoss(channels=1)

    def forward(self, input):
        coarse_denoise = self.CDN(input)
        coarse_grad, coarse_grad_x, coarse_grad_y = self.get_grad(input - coarse_denoise)
        noise_res = self.DNN(input, coarse_grad)
        out = input - noise_res
        return noise_res, coarse_denoise, coarse_grad_x, coarse_grad_y, out

    def _loss(self, Y_en, high):

        noise_res, coarse_denoise, coarse_grad_x, coarse_grad_y, out = self(Y_en)

        l2_loss = self.l2_loss(out, high)

        _, clear_x, clear_y = self.get_grad(high)
        _, x_grad, y_grad = self.get_grad(out)
        gradient_loss = self.l1_loss(x_grad, clear_x) + self.l1_loss(y_grad, clear_y)
        coarse_gradient_loss = self.l1_loss(coarse_grad_x, clear_x) + self.l1_loss(coarse_grad_y, clear_y)

        loss = l2_loss + 0.1 * gradient_loss + 0.1 * coarse_gradient_loss

        return loss
