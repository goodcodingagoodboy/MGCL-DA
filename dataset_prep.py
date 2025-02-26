# -*- coding: UTF-8 -*-
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset
import numpy as np


class Dataset_MMD(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        Args:
            data (numpy.ndarray): Data array.
            labels (numpy.ndarray): Labels array.
            transform (callable, optional): Optional transformation function.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the data and labels
        data = self.data[idx]
        label = self.labels[idx]

        # Convert to Tensor
        data = torch.from_numpy(data)
        label = torch.tensor(label)

        # Return a dictionary
        sample = {'data': data, 'label': label}
        return sample