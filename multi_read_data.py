import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os

batch_w = 600
batch_h = 400


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.img_dir = img_dir
        self.task = task
        self.train_low_data_names = []
        self.train_high_data_names = []

        self.gt_data_names = []
        self.eval_data_names = []

        if self.task == 'test':
            for root, dirs, names in os.walk(self.img_dir):
                for name in names:
                    self.train_low_data_names.append(os.path.join(root, name))

            self.train_low_data_names.sort()
            self.count = len(self.train_low_data_names)
            transform_list = []
            transform_list += [transforms.ToTensor()]
            # transform_list += [transforms.Resize([256, 256])]
            self.transform = transforms.Compose(transform_list)

        elif self.task == 'train':
            self.low_img_dir = os.path.join(self.img_dir, 'low')
            for low_root, dirs, names in os.walk(self.low_img_dir):
                high_root = os.path.join(self.img_dir, 'high')
                for name in names:
                    self.train_low_data_names.append(os.path.join(low_root, name))
                    # if '_' in name:
                    #     name = name.split('_')[0] + '.' + name.split('.')[-1]
                    # else:
                    #     name = name
                    self.train_high_data_names.append(os.path.join(high_root, name))

            self.train_low_data_names.sort()
            self.train_high_data_names.sort()
            self.count = len(self.train_low_data_names)
            transform_list = []
            transform_list += [transforms.ToTensor()]
            transform_list += [transforms.Resize([256, 256])]
            self.transform = transforms.Compose(transform_list)

        elif self.task == 'eval':
            self.gt_img_dir = self.img_dir[0]
            for gt_root, dirs, names in os.walk(self.gt_img_dir):
                eval_root = self.img_dir[1]
                for name in names:
                    self.gt_data_names.append(os.path.join(gt_root, name))
                    self.eval_data_names.append(os.path.join(eval_root, name.split('.')[0] + '.png'))

            # self.gt_data_names.sort()
            # self.eval_data_names.sort()
            self.count = len(self.gt_data_names)
            transform_list = []
            transform_list += [transforms.ToTensor()]

            self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))

        # data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)
        # im = (np.asarray(im) / 255.0)
        # img_norm = torch.from_numpy(im).float()
        return img_norm

    def __getitem__(self, index):


        if self.task == 'train':
            low = self.load_images_transform(self.train_low_data_names[index])
            low = np.asarray(low, dtype=np.float32)
            low = np.transpose(low[:, :, :], (2, 0, 1))
            img_name = self.train_low_data_names[index].split('\\')[-1]

            high = self.load_images_transform(self.train_high_data_names[index])
            high = np.asarray(high, dtype=np.float32)
            high = np.transpose(high[:, :, :], (2, 0, 1))
            high_name = self.train_high_data_names[index].split('\\')[-1]

            return torch.from_numpy(low), torch.from_numpy(high), img_name, high_name

        elif self.task == 'test':
            low = self.load_images_transform(self.train_low_data_names[index])
            low = np.asarray(low, dtype=np.float32)
            low = np.transpose(low[:, :, :], (2, 0, 1))
            img_name = self.train_low_data_names[index].split('\\')[-1]
            return torch.from_numpy(low), img_name

        elif self.task == 'eval':
            gt_data = self.load_images_transform(self.gt_data_names[index])
            gt_data = np.asarray(gt_data, dtype=np.float32)
            gt_data = np.transpose(gt_data[:, :, :], (2, 0, 1))
            gt_name = self.gt_data_names[index].split('\\')[-1]

            eval_data = self.load_images_transform(self.eval_data_names[index])
            eval_data = np.asarray(eval_data, dtype=np.float32)
            eval_data = np.transpose(eval_data[:, :, :], (2, 0, 1))
            eval_name = self.eval_data_names[index].split('\\')[-1]
            return torch.from_numpy(gt_data), torch.from_numpy(eval_data), gt_name, eval_name


    def __len__(self):
        return self.count
