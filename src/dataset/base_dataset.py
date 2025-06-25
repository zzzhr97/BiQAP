import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import json
import os.path as osp

from torch.utils.data import Dataset
from src.utils.config import INFO, DEBUG

def pad_size_fn_given(n1p, n2p):
    def pad_size_given_pair(n1, n2):
        return n1p, n2p
    return pad_size_given_pair

class BaseDataset(Dataset):
    def __init__(self, name, mode):
        self.name = name
        self.mode = mode
        
    def shuffle(self):
        INFO('Shuffle train dataloader.')
        random.shuffle(self.data)
    
    def to(self, device):
        self.device = device
        if self.data is not None:
            for data in self.data:
                data.update({
                    "F1": data['F1'].to(self.device),
                    "F2": data['F2'].to(self.device),
                    "Kp": data['Kp'].to(self.device),
                    "gt_X": data['gt_X'].to(self.device),
                    "gt_obj": data['gt_obj'].to(self.device),
                })
                
    def _load_data(self):
        self.data = []
        for f in tqdm(self.files, desc='loading data'):
            self.data.append(self._load_and_process_file(f))
        self.data = [d for d in self.data if d is not None]
                
    def _load_and_process_file(self, f):
        try:
            with open(osp.join(self.data_path, f), 'r') as tf:
                data = json.load(tf)
            return self._process_item(data)
        except Exception as e:
            tqdm.write(f"Error loading {f}: {e}")
            return None
                
    def __len__(self):
        if self.data is None:
            return len(self.files)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.data is not None:
            data = self.data[idx]
        else:
            with open(osp.join(self.data_path, self.files[idx]), 'r') as tf:
                data = json.load(tf)
            data = self._process_item(data)
        
        return data
                
    def _process_item(self, item):
        item.update({
            "F1": torch.tensor(item['F1'], dtype=torch.float32),
            "F2": torch.tensor(item['F2'], dtype=torch.float32),
            "Kp": torch.tensor(item['Kp'], dtype=torch.float32),
            "gt_X": torch.tensor(item['gt_X'], dtype=torch.float32),
            "gt_obj": torch.tensor(item['gt_obj'], dtype=torch.float32),
        })
        
        try: del item['gt_permute'] 
        except: pass
        assert item['F1'].shape[0] == item['n1'], DEBUG(f"F1.shape[0] {item['F1'].shape[0]} != n1 {item['n1']}")
        assert item['F2'].shape[0] == item['n2'], DEBUG(f"F2.shape[0] {item['F2'].shape[0]} != n1 {item['n2']}")
        assert item['Kp'].shape == (item['n1'], item['n2']), DEBUG(f'Kp.shape {item["Kp"].shape} != (n1, n2) {item["n1"], item["n2"]}')
        return item  
    
    def pad_data(self, data, pad_size_fn=None):
        """Pad data by zero."""
        if pad_size_fn is None: return data
        F1, F2, Kp, gt_X = data['F1'], data['F2'], data['Kp'], data['gt_X']
        n1, n2 = data['gt_X'].shape
        n1p, n2p = pad_size_fn(n1, n2)
        pad_size_F1 = (0, n1p-n1, 0, n1p-n1)
        pad_size_F2 = (0, n2p-n2, 0, n2p-n2)
        pad_size_Kp = (0, n2p-n2, 0, n1p-n1)
        pad_size_gt_X = pad_size_Kp
        F1p = F.pad(F1, pad_size_F1, 'constant', 0)
        F2p = F.pad(F2, pad_size_F2, 'constant', 0)
        Kpp = F.pad(Kp, pad_size_Kp, 'constant', 0)
        gt_Xp = F.pad(gt_X, pad_size_gt_X, 'constant', 0)
        gt_Xp = self.fill_Xp(gt_Xp, n1, n2, n1p, n2p)
        
        assert torch.all(F1p[:n1, :n1] == F1), "F1 is not padded correctly."
        assert torch.all(F2p[:n2, :n2] == F2), "F2 is not padded correctly."
        assert torch.all(Kpp[:n1, :n2] == Kp), "Kp is not padded correctly."
        assert torch.all(gt_Xp[:n1, :n2] == gt_X), "gt_X is not padded correctly."
                
        data.update({
            "n1": n1p,
            "n2": n2p,
            "n1_init": n1,
            "n2_init": n2,
            "F1": F1p,
            "F2": F2p,
            "Kp": Kpp,
            "gt_X": gt_Xp,
            "gt_obj": data['gt_obj'],
        })
        return data
    
    def fill_Xp(self, Xp, n1, n2, n1p, n2p):
        """Fill Xp by 1 in the padding row."""
        column_sums = Xp.sum(dim=0, keepdim=True)  # (1, n2p)
        available_columns = (column_sums == 0).nonzero(as_tuple=True)[1]  
        new_rows_count = n1p - n1
        selected_columns = available_columns[:new_rows_count]
        row_indices = torch.arange(n1, n1p, device=Xp.device)
        Xp[row_indices, selected_columns] = 1
        
        assert torch.all(Xp.sum(dim=1) == 1), "Row sum is not correct."
        assert torch.all(Xp.sum(dim=0) <= 1), "Column sum exceeds 1."
        return Xp
    
    def collate_fn(self, batch):        
        try: 
            n1 = max([item['n1'] for item in batch])
            n2 = max([item['n2'] for item in batch])
        except:
            print(batch)
        gt_obj = torch.tensor([item['gt_obj'] for item in batch])
        
        F1 = torch.stack([item['F1'] for item in batch])
        F2 = torch.stack([item['F2'] for item in batch])
        Kp = torch.stack([item['Kp'] for item in batch])
        gt_X = torch.stack([item['gt_X'] for item in batch])
        
        return {
            "n1": n1,
            "n2": n2,
            "F1": F1,
            "F2": F2,
            "Kp": Kp,
            "gt_X": gt_X,
            "gt_obj": gt_obj,
        }
        
        
        