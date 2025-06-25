from torch.utils.data import Dataset, DataLoader

from src.utils.logger import INFO, DEBUG
from .large_dataset import LargeDataset
from .qap_dataset import QAPLIBDataset
from .gm_dataset import GMDataset
from .tsp_dataset import TSPDataset
from .ged_dataset import GEDDataset


def get_dataset(dataset_cfg, device=None):
    _cfg = dataset_cfg
    try:
        dataset_module = eval(_cfg['module'])
    except:
        DEBUG(f"Module {_cfg['module']} not supported")
        raise ValueError(f"Module {_cfg['module']} not supported")
    
    problem = get_cfg(_cfg, 'problem', 'KBQAP')
    train_length = get_cfg(_cfg, 'train_length', None)
    eval_length = get_cfg(_cfg, 'eval_length', None)
    datacls = get_cfg(_cfg, 'class', None)
    batch_size = get_cfg(_cfg, 'batch_size', 1)
    
    INFO(f"Batch size: {batch_size}")
    
    if train_length is None or train_length > 0:
        train_dataset = dataset_module(
            _cfg['name'], 
            problem=problem,
            length=train_length,
            mode='train', 
            datacls=datacls,
            batch_size=batch_size,
            preload=_cfg['preload'])
        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=_cfg['num_workers'],
            shuffle=True,
            device=device)
    else:
        train_dataset, train_dataloader = None, None
    test_dataset = dataset_module(
        _cfg['name'], 
        problem=problem,
        length=eval_length,
        mode='test', 
        datacls=datacls,
        preload=_cfg['preload'])
    test_dataloader = get_dataloader(
        test_dataset,
        batch_size=1,
        num_workers=_cfg['num_workers'],
        shuffle=False,
        device=device)
    return train_dataset, test_dataset, train_dataloader, test_dataloader

def get_cfg(cfg, name_str, default=None):
    try:
        return cfg[name_str]
    except:
        return default

def get_dataloader(dataset, batch_size=1, num_workers=1, fix_seed=True, shuffle=False, device='cpu'):
    if batch_size > 1:
        collate_fn = dataset.collate_fn
    else:
        collate_fn = None
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=collate_fn,
        # pin_memory=True, 
    )