import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

from CAN_model import *
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("GILLIE")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--train_data_path', type=str, default='./data/train', help='the root folder of training dataset')
parser.add_argument('--val_data_path', type=str, default='./data/val', help='the root folder of val dataset')
parser.add_argument('--save', type=str, default='EXP/Train_CAN/', help='location of the data corpus')
parser.add_argument('--pretrained_model', type=str, default='', help='location of the data corpus')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


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
    return image[:, :, : h, : w]


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    model = CAN_Network()

    model = model.cuda()
    if args.pretrained_model != '':
        model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cuda')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)

    train_low_data_names = args.train_data_path
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    test_low_data_names = args.train_data_path
    TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        num_workers=0, shuffle=True, generator=torch.Generator(device='cuda'))

    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        num_workers=0, shuffle=True, generator=torch.Generator(device='cuda'))

    total_step = 0

    for epoch in range(1, args.epochs + 1):

        losses = []
        model.train()
        for batch_idx, (input, input_high, name, high_name) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()
            input_high = Variable(input_high, requires_grad=False).cuda()

            optimizer.zero_grad()
            loss = model._loss(input, input_high)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            losses.append(loss.item())
            logging.info('train-epoch %03d %03d %f', epoch, batch_idx, loss)

        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        # if epoch % 10 == 0:
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))




if __name__ == '__main__':
    main()
