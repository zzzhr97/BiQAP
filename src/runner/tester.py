import os
import os.path as osp
import sys
import torch
import numpy as np
import time
import math
from tqdm import tqdm

from src.dataset import get_dataset
from src.utils.path import get_project_path
from src.utils.logger import INFO, DEBUG

from .base_runner import BaseRunner


PROJECT_PATH = get_project_path(3)

class Tester(BaseRunner):
    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        if self.cfg['mode'] != 'test':
            assert False, DEBUG(f'Mode Error: {self.cfg["mode"]}')
            
        # data
        (self.train_dataset, self.test_dataset, self.train_dataloader, 
        self.test_dataloader) = get_dataset(self.cfg['dataset'], device=self.device)
        self.maximize = self.test_dataset.maximize
        
    def run(self):
        self.test_once(self.test_dataset.name, self.test_dataloader)
        
    def end(self):
        ppath = self.cfg['model']['pretrained_path']
        if ppath is not None:
            os.makedirs(self.model_dir, exist_ok=True)
            os.system(f'cp {ppath} {self.model_dir}/')
        INFO('Ending.')
        os.system(f'mv {self.result_dir} {self.result_dir}_complete')