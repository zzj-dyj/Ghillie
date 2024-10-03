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
from pytorch_msssim import SSIM

class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention_maxpool(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_maxpool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out+max_out
        return self.sigmoid(out)

class SpatialAttention_averagepool(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_averagepool, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BlancedAttention_CAM_SAM_ADD(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(BlancedAttention_CAM_SAM_ADD, self).__init__()

        self.ca = ChannelAttention_maxpool(in_planes, reduction)
        self.sa = SpatialAttention_averagepool()

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out=ca_ch.mul(sa_ch)*x

        return out

class Light_Modulation_Net(nn.Module):

    def __init__(self, num_light=32):
        super(Light_Modulation_Net, self).__init__()

        self.numl = num_light
        number_f = 32
        self.in_conv = Convblock(1, number_f)
        self.BAM1 = BlancedAttention_CAM_SAM_ADD(number_f)

        self.e_conv2 = Convblock(number_f, number_f*2)
        self.BAM2 = BlancedAttention_CAM_SAM_ADD(number_f*2)

        self.e_conv3 = Convblock(number_f*2, number_f*4)
        self.BAM3 = BlancedAttention_CAM_SAM_ADD(number_f*4)

        self.e_conv4 = Convblock(number_f*4, number_f*4)
        self.BAM4 = BlancedAttention_CAM_SAM_ADD(number_f*4)

        self.e_conv5 = Convblock(number_f * 8, number_f*2)
        self.e_conv6 = Convblock(number_f * 4, number_f)
        self.e_conv7 = Convblock(number_f * 2, number_f)

        self.out_conv = nn.Conv2d(number_f, self.numl, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.BAM1(x1)

        x2 = self.e_conv2(x1)
        x2 = self.BAM2(x2)

        x3 = self.e_conv3(x2)
        x3 = self.BAM3(x3)

        x4 = self.e_conv4(x3)
        x4 = self.BAM4(x4)

        x5 = self.e_conv5(torch.cat([x3, x4], 1))
        x6 = self.e_conv6(torch.cat([x2, x5], 1))
        x7 = self.e_conv7(torch.cat([x1, x6], 1))

        x_r = F.tanh(self.out_conv(x7))

        return x_r

class DGI_reconstruction(nn.Module):
    def __init__(self, g_factor=0.5):
        super(DGI_reconstruction, self).__init__()

        self.g_factor = g_factor

    def forward(self, y, patterns, batchsize, M, W, H):
        out = y.view(batchsize, M)
        patterns = patterns.view(M, W * H)
        one = torch.ones(M, 1).cuda()
        comput1 = torch.transpose(patterns - torch.matmul(one, torch.mean(patterns, 0).view(1, W * H)), 0, 1)
        comput2 = torch.sum(patterns, 1)

        gamma = self.g_factor * torch.mean(out, 1) / torch.mean(comput2)
        gamma = gamma.view(batchsize, 1)
        comput2 = comput2.view(1, M)
        temp = gamma * comput2

        temp = torch.transpose(out - temp, 0, 1)
        temp = temp - torch.mean(temp, 0)

        DGI = torch.matmul(comput1, temp)

        DGI = (DGI - torch.min(DGI)) / (torch.max(DGI) - torch.min(DGI))
        DGI = DGI.view(batchsize, 1, W, H)
        # print(DGI.shape)
        return DGI

class AMSSIMLoss(nn.Module):
    def __init__(self, kernal_size=11, num_channels=1, C=9e-4, device='cuda:0'):
        super(AMSSIMLoss, self).__init__()
        self.kernal_size = kernal_size
        self.avg_kernal = torch.ones(num_channels, 1, self.kernal_size, self.kernal_size) / (self.kernal_size) ** 2
        self.avg_kernal = self.avg_kernal.to(device)
        self.c = C

    def forward(self, enhance_img, input, input_high):

        input_high_total_intensity = input_high.mean()

        batch_size, num_channels = input.shape[0], input.shape[1]

        low_images_mean = F.conv2d(input, self.avg_kernal, stride=self.kernal_size, groups=num_channels)

        low_images_var = torch.abs(F.conv2d(input ** 2, self.avg_kernal, stride=self.kernal_size,
                                            groups=num_channels) - low_images_mean ** 2)

        high_images_mean = F.conv2d(input_high, self.avg_kernal, stride=self.kernal_size, groups=num_channels)

        high_images_var = torch.abs(F.conv2d(input_high ** 2, self.avg_kernal, stride=self.kernal_size,
                                            groups=num_channels) - high_images_mean ** 2)

        enhance_images_mean = F.conv2d(enhance_img, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
        enhance_images_var = torch.abs(F.conv2d(enhance_img ** 2, self.avg_kernal, stride=self.kernal_size,
                                               groups=num_channels) - enhance_images_mean ** 2)

        low_enhance_images_var = F.conv2d(input * enhance_img, self.avg_kernal, stride=self.kernal_size,
                                         groups=num_channels) - low_images_mean * enhance_images_mean
        high_enhance_images_var = F.conv2d(input_high * enhance_img, self.avg_kernal, stride=self.kernal_size,
                                         groups=num_channels) - high_images_mean * enhance_images_mean

        C = torch.ones_like(enhance_images_mean) * self.c

        ssim_l_low_enhance = (2 * low_images_mean * enhance_images_mean + C) / \
                            (low_images_mean ** 2 + enhance_images_mean ** 2 + C)
        ssim_l_high_enhance = (2 * high_images_mean * enhance_images_mean + C) / \
                            (high_images_mean ** 2 + enhance_images_mean ** 2 + C)

        ssim_s_low_enhance = (2 * low_enhance_images_var + C) / (low_images_var + enhance_images_var + C)
        ssim_s_high_enhance = (2 * high_enhance_images_var + C) / (high_images_var + enhance_images_var + C)

        score_low_high_enhance = (low_images_mean > input_high_total_intensity) * ssim_s_low_enhance * ssim_l_low_enhance + (low_images_mean <= input_high_total_intensity) * ssim_s_high_enhance * ssim_l_high_enhance

        ssim_loss = 1 - score_low_high_enhance.mean()
        #print('ssim_loss:', ssim_loss)
        return ssim_loss

class Network(nn.Module):

    def __init__(self, batchsize=1, M=10):
        super(Network, self).__init__()
        self.batchsize = batchsize
        self.M = M
        self.LMN = Light_Modulation_Net(num_light=M)
        self.DGIR = DGI_reconstruction(1)

        self.l1_loss = nn.L1Loss()
        self.amssim_loss = AMSSIMLoss(kernal_size=11, num_channels=1, C=9e-4, device='cuda:0')
        # self.ssim = SSIM(data_range=1., size_average=True, channel=1)

    def forward(self, input):

        self.y_low, Cb, Cr = torch.split(kornia.color.rgb_to_ycbcr(input), 1, dim=1)

        Str_light = self.LMN(self.y_low)
        Str_light_r = Str_light
        num = 1
        for _ in range(num - 1):
            Str_light_r = torch.cat([Str_light_r, Str_light], dim=1)

        Str_light_r = Str_light_r.permute(1, 0, 2, 3)
        W, H = self.y_low.shape[2], self.y_low.shape[3]

        temp = F.conv2d(self.y_low, Str_light_r, stride=1)
        mean = torch.mean(temp)
        var = torch.mean(torch.pow(temp - mean, 2.0))
        temp = (temp - mean) / torch.pow(var, 0.5)
        temp = temp.permute(0, 2, 3, 1)

        y_new = self.DGIR(temp, Str_light_r, self.batchsize, self.M * num, W, H)

        image_out = kornia.color.ycbcr_to_rgb(torch.cat([y_new, Cb, Cr], dim=1))

        return image_out, y_new, Str_light

    def _loss(self, input, input_high):
        _, Y_out, Str_light = self(input)
        Y_high, _, _ = torch.split(kornia.color.rgb_to_ycbcr(input_high), 1, dim=1)

        l1_loss = self.l1_loss(Y_high, Y_out)
        # ssim_loss = (1 - self.ssim(Y_high, Y_out))
        amssim_loss = self.amssim_loss(Y_out, self.y_low, Y_high)

        loss = l1_loss + amssim_loss

        return loss





