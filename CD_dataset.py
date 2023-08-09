import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tr
import os
import numpy as np
import random
from skimage import io
from tqdm import tqdm as tqdm
from math import ceil
import time
from itertools import chain
import time

class RandomFlip(object):
    """Flip randomly the images in a sample."""
    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']
        
        if random.random() > 0.5:
            I1 =  I1.numpy()[:,:,::-1].copy()
            I1 = torch.from_numpy(I1)
            I2 =  I2.numpy()[:,:,::-1].copy()
            I2 = torch.from_numpy(I2)
            label =  label.numpy()[:,::-1].copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}

class RandomRot(object):
    """Rotate randomly the images in a sample."""
    def __call__(self, sample):
        I1, I2, label = sample['I1'], sample['I2'], sample['label']
        
        n = random.randint(0, 3)
        if n:
            I1 =  sample['I1'].numpy()
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I1 = torch.from_numpy(I1)
            I2 =  sample['I2'].numpy()
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I2 = torch.from_numpy(I2)
            label =  sample['label'].numpy()
            label = np.rot90(label, n, axes=(0, 1)).copy()
            label = torch.from_numpy(label)

        return {'I1': I1, 'I2': I2, 'label': label}

def read_optical_img(path): #path include imgname, A/B, and savepath
    img = io.imread(path)

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    I = np.stack((r,g,b),axis=2).astype('float')

    I = (I - I.mean()) / I.std()
    return I

def read_optical_img_trio(path, name):
    """Read cropped Sentinel-2 image pair and change map."""
    #  read images
    I1 = read_optical_img(path + '/A/' + name)
    I2 = read_optical_img(path + '/B/' + name)

    cm = io.imread(path + '/label/' + name, as_gray=True) 
    cm=cm!= 0
    return I1, I2, cm

def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)

class ChangeDetectionDataset(Dataset):
    def __init__(self, path, patch_size = 224, stride = 224, transform=None, FP_MODIFIER = 10):
        self.transform = transform
        self.path = path
        self.patch_size = patch_size
        self.stride = stride

        self.names = []
        
        txt_name = path + 'img.txt'
        f = open(txt_name, "r")
        lines = f.readlines()
        f.close()
        
        for line in lines:
            if 'jpg'in line or 'bmp' in line or 'png' in line:
                name = line.strip()
                # name = name[2:]#############A/*.jpg=>*.jpg
                self.names.append(name)
        
        n_pix = 0
        true_pix = 0
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            I1, I2, cm = read_optical_img_trio(self.path, im_name)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()
            
            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            v1 = (s[1] - self.patch_size ) / self.stride + 1.0
            v2 = (s[2] - self.patch_size ) / self.stride + 1.0
            n1 = ceil(v1)
            n2 = ceil(v2)

            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate path coordinates
            for i in range(n1):
                si = self.stride*i
                ei = self.stride*i + self.patch_size
                if ei > s[1]:
                    ei = s[1]
                    si = ei - self.patch_size
                for j in range(n2):
                    sj = self.stride*j
                    ej = self.stride*j + self.patch_size
                    if ej > s[2]:
                        ej = s[2]
                        sj = ej - self.patch_size
                    current_patch_coords = (im_name, 
                                    [si, ei, sj, ej],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    self.patch_coords.append(current_patch_coords)
                    
        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]
        print(true_pix, n_pix)
        
    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1*np.array(label)).float()
        
        sample = {'I1': I1, 'I2': I2, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class TestDataset(Dataset):
    def __init__(self, path, patch_size = 224):
        self.path = path
        self.patch_size = patch_size
        
        self.names = []
        
        txt_name = path + 'img.txt'
        f = open(txt_name, "r")
        lines = f.readlines()
        f.close()

        for line in lines:
            if 'jpg'in line or 'bmp' in line or 'png' in line:
                name = line.strip()
                # name = name[2:]
                self.names.append(name)
        
        n_pix = 0
        true_pix = 0
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            I1, I2, cm = read_optical_img_trio(self.path, im_name)
                
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()
            
            s = self.imgs_1[im_name].shape

            n1 = s[1] // self.patch_size
            if s[1] % self.patch_size > 0:
                n1 = n1 + 1
            
            n2 = s[2] // self.patch_size
            if s[2] % self.patch_size > 0:
                n2 = n2 + 1

            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i

            for i in range(n1):
                start_i = i * self.patch_size
                end_i = min((i + 1) * self.patch_size, s[1])
                for j in range(n2):
                    start_j = j * self.patch_size
                    end_j = min((j + 1) * self.patch_size, s[2])

                    current_patch_coords = (im_name, 
                                    [start_i, end_i, start_j, end_j])
                    self.patch_coords.append(current_patch_coords)
        print(n_pix)
        
    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        
        h = limits[1] - limits[0]
        w = limits[3] - limits[2]

        if h < self.patch_size or w < self.patch_size:
            start_h = limits[0] - (self.patch_size - h)
            start_w = limits[2] - (self.patch_size - w)
            I1 = self.imgs_1[im_name][:, start_h:limits[1], start_w:limits[3]]
            I2 = self.imgs_2[im_name][:, start_h:limits[1], start_w:limits[3]]
            
            label = self.change_maps[im_name][start_h:limits[1], start_w:limits[3]]
            label = torch.from_numpy(1*np.array(label)).float()
        else:
            I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
            I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
            
            label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
            label = torch.from_numpy(1*np.array(label)).float()

        sample = {'I1': I1, 'I2': I2, 'label': label}

        return sample, current_patch_coords


