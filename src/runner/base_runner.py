import torch
import torch.nn.functional as F
import time
import os
from tqdm import tqdm

from src.utils.path import get_project_path
from src.utils.config import Config
from src.utils.logger import INFO, DEBUG

from src.utils.loss import get_loss
from src.models import get_model
from src.utils.post_process import get_post_process
from src.utils.problem import eval_obj
from src.utils.state import STATE
from src.utils.utils import set_seed


class BaseRunner(object):
    def __init__(self, cfg_path):
        # config
        self.config = Config(cfg_path)
        self.cfg = self.config.get_cfg()
        self.model_dir = self.config.get_model_dir()
        self.result_dir = self.config.get_result_dir()
        self.problem = self.cfg['dataset']['problem']
        
        # seed
        set_seed(self.cfg['random_seed'])
            
        # device
        gpu_id = self.cfg['gpu']
        self.device = torch.device(f'cuda:{gpu_id}') if gpu_id > -1 and torch.cuda.is_available() else torch.device('cpu')
        
        # loss
        self.loss_fn = get_loss(self.cfg['loss'])
        
        # model
        self.model = get_model(self.cfg['model'], self.cfg['model_setting'], self.device)
        
        # post-process
        self.pp = get_post_process(self.cfg['post_process'])
        
    def test_once(self, name, dataloader):
        INFO('Start testing...')
        start_t = time.time()
        self.eval_status()
        
        eval_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        obj_res = .0
        obj_pr = .0
        obj_gt = .0
        obj_mae = .0  
        obj_mse = .0 
        acc = .0
        better = 0
        
        length = len(dataloader)   
        
        try:
            with torch.no_grad():
                with tqdm(total=length, desc=f'[Eval {name}]') as pbar:
                    for i, data in enumerate(dataloader):
                        data = self.process_data(data)
                        
                        st = time.time()
                        
                        pred, pred_X = self.forward_data(data, forward_time=self.cfg['eval']['forward_time'])
                        
                        t = time.time() - st
                        
                        loss = self.loss_fn(pred, data, maximize=self.maximize)
                        eval_loss += loss
                        
                        result = eval_obj(pred_X, data, self.maximize)
                        obj_res += result['res'].item()
                        obj_pr += result['pred_obj'].item()
                        obj_gt += result['gt_obj'].item()
                        obj_mae += result['mae'].item()
                        obj_mse += result['mse'].item()
                        acc += result['acc'].item()
                        if result['res'].item() <= 0:
                            better += 1
                            
                        pbar.update(1)
                        pbar.set_description_str(f'[Eval {name}] [Loss: {eval_loss/(i+1):.4f}] ' + \
                            f'[RES: {obj_res/(i+1):.4f}] ' + \
                            f'[PR/GT: {obj_pr/(i+1):.4f}/{obj_gt/(i+1):.4f}] ' + \
                            f'[MAE: {obj_mae/(i+1):.4f}] ' + \
                            f'[MSE: {obj_mse/(i+1):.4f}] ' + \
                            f'[ACC: {acc/(i+1):.4f}] ' + \
                            f'[Better: {better} / {i+1}]'
                        )
                    
                eval_loss /= length
                obj_res /= length
                obj_pr /= length
                obj_gt /= length
                obj_mae /= length
                obj_mse /= length
                acc /= length
                eval_str = f'[Eval {name}] [Loss: {eval_loss.item():.04f}] ' + \
                    f'[RES: {obj_res:.04f}] ' + \
                    f'[PR/GT: {obj_pr:.04f}/{obj_gt:.04f}] ' + \
                    f'[MAE: {obj_mae:.04f}] ' + \
                    f'[MSE: {obj_mse:.04f}] ' + \
                    f'[ACC: {acc:.04f}] ' + \
                    f'[Better: {better} / {length}] ' + \
                    f'[Time: {time.time()-start_t:.2f}s]\n'
                INFO(eval_str)
        except KeyboardInterrupt:
            DEBUG('KeyboardInterrupt')
        self.train_status()
        
    def forward_data(self, data, forward_time=1):
        total_pred, total_pred_X = [], []
        for _ in range(forward_time):
            forward_data = self.process_data(data)
            pred = self.model(forward_data)
            pred_X = self.pp(pred)
            assert len(pred.shape) == 3
            assert len(pred_X.shape) == 3
            total_pred.append(pred.detach().clone())
            total_pred_X.append(pred_X.detach().clone())
        total_pred = torch.concat(total_pred, dim=0)
        total_pred_X = torch.concat(total_pred_X, dim=0)
        return total_pred, total_pred_X
        
    def process_data(self, data):
        processed = {**data}
        processed.update({
            "sbs": self.cfg['model_setting']['sample_batch_size'],
        })
        if self.problem == 'KBQAP':
            F1, F2, Kp = data['F1'], data['F2'], data['Kp']
            if STATE["train"]:
                F1.requires_grad = True
                F2.requires_grad = True
                Kp.requires_grad = True
            else:
                F1.requires_grad = False
                F2.requires_grad = False
                Kp.requires_grad = False
            processed.update({
                "F1": F1.to(self.device),
                "F2": F2.to(self.device),
                "Kp": Kp.to(self.device),
                "gt_X": data['gt_X'].to(self.device),
                "gt_obj": data['gt_obj'].to(self.device)
            }) 
        elif self.problem == 'LLQAP':
            assert False, DEBUG('Not implemented.') 
 
        return processed
        
    def train_status(self):
        self.model.train()
        if 'train_setting' in self.cfg['model_setting']:
            self.model._set_param(self.cfg["model_setting"]["train_setting"])
            try:
                self.cfg['model_setting']['sample_batch_size'] = self.cfg["model_setting"]["train_setting"]["sample_batch_size"]
                INFO(f"Train: sample_batch_size {self.cfg['model_setting']['sample_batch_size']}")
            except:
                pass
            
        STATE["train"] = True
            
    def eval_status(self):
        self.model.eval()
        if 'test_setting' in self.cfg['model_setting']:
            self.model._set_param(self.cfg["model_setting"]["test_setting"])
            try:
                self.cfg['model_setting']['sample_batch_size'] = self.cfg["model_setting"]["test_setting"]["sample_batch_size"]
                INFO(f"Test: sample_batch_size {self.cfg['model_setting']['sample_batch_size']}")
            except:
                pass
            
        STATE["train"] = False
            
    def id(self):
        return self.result_dir + '_complete'