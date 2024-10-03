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

from pytorch_msssim import MS_SSIM, SSIM
class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Quality_Adjustment_Net(nn.Module):

    def __init__(self, input_channel, output_channel):
        super(Quality_Adjustment_Net, self).__init__()

        number_f = 32
        self.in_conv = Convblock(input_channel, number_f)
        self.e_conv2 = Convblock(number_f, number_f * 2)
        self.e_conv3 = Convblock(number_f * 2, number_f * 4)
        self.e_conv4 = Convblock(number_f * 4, number_f * 4)
        self.e_conv5 = Convblock(number_f * 8, number_f * 2)
        self.e_conv6 = Convblock(number_f * 4, number_f)
        self.e_conv7 = Convblock(number_f * 2, number_f)

        self.out_conv = nn.Conv2d(number_f, output_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.in_conv(x)
        # p1 = self.maxpool(x1)
        x2 = self.e_conv2(x1)
        # p2 = self.maxpool(x2)
        x3 = self.e_conv3(x2)
        # p3 = self.maxpool(x3)
        x4 = self.e_conv4(x3)
        # x4 = self.upsample(x4)

        x5 = self.e_conv5(torch.cat([x3, x4], 1))
        # x5 = self.upsample(x5)
        x6 = self.e_conv6(torch.cat([x2, x5], 1))
        # x6 = self.upsample(x6)
        x7 = self.e_conv7(torch.cat([x1, x6], 1))
        x_r = F.tanh(self.out_conv(x7))

        return x_r

class MSSSIML1Loss(nn.Module):
    def __init__(self, channels):
        super(MSSSIML1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ms_ssim = MS_SSIM(data_range=1., size_average=True, channel=channels)
        self.alpha = 1.0

    def forward(self, output, target):
        ms_ssim = 1 - self.ms_ssim(output, target)
        l1_loss = self.l1_loss(output, target)
        loss = l1_loss + self.alpha * ms_ssim
        return loss


class CAN_Network(nn.Module):

    def __init__(self):
        super(CAN_Network, self).__init__()
        self.CAN = Quality_Adjustment_Net(4, 2)
        self.Cb_l1 = nn.L1Loss()
        self.Cr_l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=1., size_average=True, channel=3)
        # self.msssim = MSSSIML1Loss(channels=3)

    def forward(self, input, y_high):

        self.y_low, Cb_low, Cr_low = torch.split(kornia.color.rgb_to_ycbcr(input), 1, dim=1)

        x = torch.cat([self.y_low, Cb_low, Cr_low, y_high], dim=1)
        out = self.CAN(x)

        return out

    def _loss(self, input, input_high):

        y_high, Cb_high, Cr_high = torch.split(kornia.color.rgb_to_ycbcr(input_high), 1, dim=1)
        out = self(input, y_high)
        pred_cb, pred_cr = torch.split(out, 1, dim=1)

        Cb_l1 = self.Cb_l1(Cb_high, pred_cb)
        Cr_l1 = self.Cr_l1(Cr_high, pred_cr)

        image_pred = kornia.color.ycbcr_to_rgb(torch.cat([y_high, pred_cb, pred_cr], dim=1))
        ssim = 1 - self.ssim(input_high, image_pred)

        loss = Cb_l1 + Cr_l1 + ssim

        return loss


