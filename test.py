import os
import sys
import numpy as np
import torch
import time
import argparse
import torch.utils
from math import exp
import math
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from LMN_model import *
from DNN_model import DNN_Network
from CAN_model import *
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("Ghillie")
parser.add_argument('--data_path', type=str, default='./data/val', help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./results', help='location of the data corpus')
parser.add_argument('--LMN_model', type=str, default='./weights/LMN_weights.pt', help='location of the data corpus')
parser.add_argument('--DNN_model', type=str, default='./weights/DNN_weights.pt', help='location of the data corpus')
parser.add_argument('--CAN_model', type=str, default='./weights/CAN_weights.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--M', type=int, default=32, help='number of structured light')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def padding(image, divide_size=4):
    n, c, h, w = image.shape
    padding_h = divide_size - h % divide_size
    padding_w = divide_size - w % divide_size
    image = F.pad(image, (0, padding_w, 0, padding_h), "reflect")
    return image, h, w

def unpadding(image, h, w):
    return image[:, :, :h, :w]

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    LMN_model = Network(args.batch_size, args.M)
    LMN_model.load_state_dict(torch.load(args.LMN_model))
    LMN_model = LMN_model.cuda()

    DNN_model = DNN_Network()
    DNN_model.load_state_dict(torch.load(args.DNN_model))
    DNN_model = DNN_model.cuda()

    CAN_model = CAN_Network()
    CAN_model.load_state_dict(torch.load(args.CAN_model))
    CAN_model = CAN_model.cuda()


    with torch.no_grad():
        Runtime = []
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input, volatile=True).cuda()
            image_name = image_name[0].split('/')[-1].split('.')[0]
            # y_low, Cb, Cr = torch.split(kornia.color.rgb_to_ycbcr(input), 1, dim=1)

            start_time = time.perf_counter()
            x, y_en, s = LMN_model(input)
            _, _, _, _, y_en = DNN_model(y_en)
            out_cbcr = CAN_model(input, y_en)
            cb_out, cr_out = torch.split(out_cbcr, 1, dim=1)
            out_image = kornia.color.ycbcr_to_rgb(torch.cat([y_en, cb_out, cr_out], dim=1))
            end_time = time.perf_counter()

            Runtime.append(end_time - start_time)

            u_name = image_name + '.png'
            print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(out_image, u_path)

        Runtime_mean = sum(Runtime) / len(Runtime)
        Runtime_variance = sum((x - Runtime_mean) ** 2 for x in Runtime) / len(Runtime)

        print(f"Mean of runtime: {Runtime_mean}")
        print(f"Variance of runtime: {Runtime_variance}")


if __name__ == '__main__':
    main()
