import os
import os.path as osp
import sys
import torch
import numpy as np
import time
import math
from tqdm import tqdm, trange

from src.utils.path import get_project_path
from src.utils.logger import INFO, DEBUG

from src.dataset import get_dataset
from src.utils.optimizer import get_optimizer
from src.utils.scheduler import get_scheduler
from src.utils.time import get_time_status
from src.utils.state import STATE

from .base_runner import BaseRunner

PROJECT_PATH = get_project_path(3)

class Trainer(BaseRunner):
    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        self.train_cfg = self.cfg['train']
        if self.cfg['mode'] != 'train':
            assert False, DEBUG(f'Mode Error: {self.cfg["mode"]}')
            
        # data
        (self.train_dataset, self.test_dataset, self.train_dataloader, 
        self.test_dataloader) = get_dataset(self.cfg['dataset'], device=self.device)
        self.maximize = self.test_dataset.maximize
        self.batch_size = self.cfg['dataset']['batch_size']
        
        # optimizer
        self.optimizer = get_optimizer(self.train_cfg['optimizer'], self.model)
        
        # scheduler
        self.scheduler = get_scheduler(self.train_cfg['scheduler'], self.optimizer)
        
        # eval
        self.print_fq = self.cfg['eval']['print_fq']
        self.eval_fq = self.cfg['eval']['eval_fq']
        self.max_save = self.cfg['eval']['max_save']
        if self.max_save is None: self.max_save = int(1e12)
        
    def run(self):
        INFO('Start training...')
        
        cumulative_size = self.train_cfg['cumulative_size']
        total_step = self.train_cfg['total_step']
        step = 0
        bstep = 0
        
        INFO(f'Cumulative size: {cumulative_size}')
        
        # init
        loss = .0
        self.optimizer.zero_grad()
        
        start_t = time.time()
        
        self.train_status()
        
        epochs = np.ceil(total_step * cumulative_size / len(self.train_dataloader)).astype('int')
        pbar = trange(total_step, ascii=True, leave=True, desc="step", position=0)
        for epoch in range(epochs):
            
            STATE['epoch'] = epoch
            
            for i, data in enumerate(self.train_dataloader):
                data = self.process_data(data)
                pred = self.model(data)
                cur_loss = self.loss_fn(pred, data, maximize=self.maximize)
                cur_loss.backward()
                loss += cur_loss.item()
                bstep += 1
                
                del data, pred
                
                if bstep >= cumulative_size:
                    bstep = 0
                    step += 1
                    epoch_frac = epoch + i / len(self.train_dataloader)
                    
                    loss = loss / cumulative_size
                    self.optimizer.step()
                    self.scheduler.step()
                    torch.cuda.empty_cache()
                    
                    pbar.update(1)
                    pbar.set_description_str(
                        f'[Train] [Epoch: {epoch_frac:.06f}] ' + \
                        f'[Step: {step}] ' + \
                        f'[Loss: {loss:.04f}]')
                    
                    if step % self.print_fq == 0:
                        ela_t, rem_t, avg_t = get_time_status(start_t, total_step, step)
                        INFO(f'[Train] [Epoch: {epoch_frac:.06f}] [Step: {step}] [Loss: {loss:.04f}] ' + \
                            f'[{ela_t} / {rem_t} | {avg_t}]')
                        
                    if step % self.eval_fq == 0:
                        self.eval_model()
                        self.save_model(step)
                        
                    # restart
                    loss = .0
                    self.optimizer.zero_grad()
                    
                    if step >= total_step: break
                    
            if step >= total_step: break
                    
        ela_t, rem_t, avg_t = get_time_status(start_t, total_step, total_step)
        INFO(f'Finish training. Total time: {ela_t}')
        
    def eval_model(self):
        self.test_once(self.test_dataset.name, self.test_dataloader)
    
    def save_model(self, step):
        model_path = osp.join(self.model_dir, f'model_{step}.pth')
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        INFO(f'[Eval] Save model with step [{step}] to {model_path}')
    
    def end(self):
        model_path = osp.join(self.model_dir, f'final.pth')
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        ppath = self.cfg['model']['pretrained_path']
        if ppath is not None:
            os.system(f'cp {ppath} {self.model_dir}/')
            
        INFO('Ending.')
        os.system(f'mv {self.result_dir} {self.result_dir}_complete')