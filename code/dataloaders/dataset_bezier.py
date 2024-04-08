import os
import cv2
import torch
import random
import copy
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from dataloaders.bezier_curve import *


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None,prob=0.8):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.prob = prob
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        elif self.split == 'test':
            with open(self._base_dir + '/test.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
                         
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if self.split == 'train':
            wimage = nonlinear_transformation(image,mode='weak',prob = self.prob)
            # simage = nonlinear_transformation(image,mode='weak',prob = self.prob)       
            simage = nonlinear_transformation(image,mode='strong')       

            # wimage = image
            # simage = image
            sample = {'wimage': wimage, 'simage':simage, 'label': label}
            sample = self.transform(sample)
        if self.split == "val":
            sample = {'image': image, 'label': label}
            
        sample["idx"] = idx
        return sample


def random_rot_flip(wimage, simage, label):
    k = np.random.randint(0, 4)
    wimage = np.rot90(wimage, k)
    simage = np.rot90(simage, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    wimage = np.flip(wimage, axis=axis).copy()
    simage = np.flip(simage, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return wimage, simage, label


def random_rotate(wimage, simage, label):
    angle = np.random.randint(-20, 20)
    wimage = ndimage.rotate(wimage, angle, order=0, reshape=False)
    simage = ndimage.rotate(simage, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return wimage, simage, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        wimage,simage, label = sample['wimage'],sample['simage'], sample['label']
        if random.random() > 0.5:
            wimage,simage, label = random_rot_flip(wimage,simage, label)
        elif random.random() > 0.5:
            wimage,simage, label = random_rotate(wimage,simage, label)
        x, y = wimage.shape
        wimage = zoom(
            wimage, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        simage = zoom(
            simage, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        wimage = torch.from_numpy(
            wimage.astype(np.float32)).unsqueeze(0)
        simage = torch.from_numpy(
            simage.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'wimage': wimage, 'simage': simage, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)