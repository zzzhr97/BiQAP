import numpy as np
from pathlib import Path
import os.path as osp
import re
import pickle
import torch
import random

from src.utils.path import get_project_path
from src.utils.config import INFO, DEBUG
from src.dataset.base_dataset import BaseDataset
from src.utils.problem import get_obj_KBQAP


PROJECT_PATH = get_project_path(3)
DATA_PATH = osp.join(PROJECT_PATH, 'data')

class QAPLIBDataset(BaseDataset):
    CLSLIST = ['bur', 'chr', 'els', 'esc', 'had', 'kra', 'lipa', 'nug', 'rou', 'scr', 'sko', 'ste', 'tai', 'tho', 'wil']
    DIR = osp.join(DATA_PATH, 'QAPLIB')
    MAX_TRAIN_SIZE = 256
    MAX_TEST_SIZE = 256
    ENABLE_SLN = True   # only the samples with solution are used
    
    def __init__(self, name='QAPLIB', mode='train', datacls=None, preload=True, **kwargs):
        assert name in ['QAPLIB'], DEBUG(f'Invalid dataset name: {name}')
        super().__init__(name, mode)
        self.mode = mode
        self.data_path = osp.join(self.DIR)
        self.qap_path = Path(osp.join(self.DIR, 'raw'))
        self.maximize = False

        if datacls is not None and datacls != 'none':
            idx = self.CLSLIST.index(datacls)
            self.CLSLIST = [self.CLSLIST[idx]]
        else:
            self.CLSLIST = self.CLSLIST

        self.data_list = []
        for inst in self.CLSLIST:
            for dat_path in self.qap_path.glob(inst + '*.dat'):
                name = dat_path.name[:-4]
                prob_size = int(re.findall(r"\d+", name)[0])
                if (self.mode == 'test' and prob_size > self.MAX_TEST_SIZE) \
                    or (self.mode == 'train' and prob_size > self.MAX_TRAIN_SIZE):
                    continue
                self.data_list.append(name)

        # remove trivial instance esc16f
        if 'esc16f' in self.data_list:
            self.data_list.remove('esc16f')

        # define compare function
        def name_cmp(a, b):
            a = re.findall(r'[0-9]+|[a-z]+', a)
            b = re.findall(r'[0-9]+|[a-z]+', b)
            for _a, _b in zip(a, b):
                if _a.isdigit() and _b.isdigit():
                    _a = int(_a)
                    _b = int(_b)
                cmp = (_a > _b) - (_a < _b)
                if cmp != 0:
                    return cmp
            if len(a) > len(b): return -1
            elif len(a) < len(b): return 1
            else: return 0

        def cmp_to_key(mycmp):
            'Convert a cmp= function into a key= function'
            class K:
                def __init__(self, obj, *args):
                    self.obj = obj
                def __lt__(self, other):
                    return mycmp(self.obj, other.obj) < 0
                def __gt__(self, other):
                    return mycmp(self.obj, other.obj) > 0
                def __eq__(self, other):
                    return mycmp(self.obj, other.obj) == 0
                def __le__(self, other):
                    return mycmp(self.obj, other.obj) <= 0
                def __ge__(self, other):
                    return mycmp(self.obj, other.obj) >= 0
                def __ne__(self, other):
                    return mycmp(self.obj, other.obj) != 0
            return K

        # sort data list according to the names
        self.data_list.sort(key=cmp_to_key(name_cmp))
        
        if preload: 
            if len(self.CLSLIST) == 15: self.preload_path = osp.join(self.DIR, f'total.pkl')
            else:
                clsstr = '_'.join(self.CLSLIST)
                self.preload_path = osp.join(self.DIR, f'{clsstr}.pkl')
            if osp.exists(osp.join(self.preload_path)):
                print(f"QAPLIB {self.CLSLIST}: loading data from {self.preload_path}")
                with open(osp.join(self.preload_path), 'rb') as f:
                    self.data = pickle.load(f)
            else:
                self._load_data()
                pickle.dump(self.data, open(osp.join(self.preload_path), 'wb'))
                print(f"QAPLIB {self.CLSLIST}: {len(self.data)} data loaded and saved to {self.preload_path}")
                
        if self.mode == 'train':
            random.shuffle(self.data)
                
    def _load_data(self):
        self.data = []
        for idx in range(len(self.data_list)):
            pair = self.get_pair(idx)
            if pair is not None:
                self.data.append(pair)
        
    def get_pair(self, idx):
        """
        Get QAP data by index
        :param idx: dataset index
        :return: (pair of data, groundtruth permutation matrix)
        """
        name = self.data_list[idx]
        
        def split_line(x):
            for _ in re.split(r'[,\s]', x.rstrip('\n')):
                if _ == "":
                    continue
                else:
                    yield int(_)

        dat_path = self.qap_path / (name + '.dat')
        dat_file = dat_path.open()
        dat_list = [[_ for _ in split_line(line)] for line in dat_file]
        prob_size = dat_list[0][0]

        # read data
        r = 0
        c = 0
        F1 = [[]]
        F2 = [[]]
        F = F1
        for l in dat_list[1:]:
            F[r] += l
            c += len(l)
            assert c <= prob_size
            if c == prob_size:
                r += 1
                if r < prob_size:
                    F.append([])
                    c = 0
                else:
                    F = F2
                    r = 0
                    c = 0
        F1 = torch.tensor(F1, dtype=torch.float32)
        F2 = torch.tensor(F2, dtype=torch.float32).transpose(0, 1).contiguous()
        assert F1.shape == F2.shape == (prob_size, prob_size)
        Kp = torch.zeros((prob_size, prob_size), dtype=torch.float32)

        try:
            sln_path = self.qap_path / (name + '.sln')
            sln_file = sln_path.open()
            sln_list = [[_ for _ in split_line(line)] for line in sln_file]
            
            perm_list = []
            for _ in sln_list[1:]:
                perm_list += _
            assert len(perm_list) == prob_size
            gt_X = torch.zeros((prob_size, prob_size), dtype=torch.float32)
            for r, c in enumerate(perm_list):
                gt_X[r, c - 1] = 1
                
            gt_obj = torch.tensor(sln_list[0][1], dtype=torch.float32)
        except:
            perm = torch.tensor(np.random.choice(np.arange(prob_size), prob_size, replace=False), dtype=torch.int64)
            gt_X = torch.zeros((prob_size, prob_size), dtype=torch.float32)
            gt_X[torch.arange(prob_size), perm] = 1
            gt_obj = get_obj_KBQAP(F1.unsqueeze(0), F2.unsqueeze(0), Kp.unsqueeze(0), gt_X.unsqueeze(0)).squeeze(0)
            # return None

        data = {
            "name": name,
            "F1": F1,
            "F2": F2,
            "Kp": Kp,
            "gt_X": gt_X,
            "gt_obj": gt_obj,
        }
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            if self.data is not None:
                data = self.data[idx]
            else:
                data = self.get_pair(idx)
        except:
            print(idx)
            print(len(self.data))
            exit()
        
        return data