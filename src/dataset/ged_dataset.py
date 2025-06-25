import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import random
import os
import os.path as osp
import json
import pickle
from tqdm import tqdm

from src.dataset.base_dataset import BaseDataset
from src.utils.path import get_project_path
from src.utils.logger import INFO, DEBUG


PROJECT_PATH = get_project_path(3)
DATA_PATH = osp.join(PROJECT_PATH, 'data')
    
class GEDDataset(BaseDataset):
    DIR = osp.join(DATA_PATH, 'GED')
    DATASET = [
        'AIDS',
        'Linux',
        'IMDB'
    ]
    MODES = ['train', 'test']
    def __init__(self, name, length=None, mode='train', preload=True, **kwargs):
        assert name in self.DATASET, f'{name} not in {self.DATASET}'
        assert mode in self.MODES, f'{mode} not in {self.MODES}'
        super().__init__(name, mode)
        self.name = name
        self.length = length
        self.data_path = osp.join(self.DIR, name, mode)
        self.files = os.listdir(self.data_path)
        self.maximize = True
        
        random.shuffle(self.files)
        if length is not None: self.files = self.files[:length]
        print(f"{self.name}: {len(self.files)} items in {self.data_path}")
        
        self.data = None
        if preload: 
            self.preload_path = osp.join(self.DIR, name, f'{mode}.pkl')
            if osp.exists(osp.join(self.preload_path)):
                print(f"{self.name}: loading data from {self.preload_path}")
                with open(osp.join(self.preload_path), 'rb') as f:
                    self.data = pickle.load(f)
            else:
                self._load_data()
                pickle.dump(self.data, open(osp.join(self.preload_path), 'wb'))
                print(f"{self.name}: data loaded and saved to {self.preload_path}")   
        if length is not None: self.data = self.data[:length]   