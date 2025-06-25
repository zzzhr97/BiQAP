import networkx as nx
import numpy as np
import os
import os.path as osp
import random
import json
import time
from datetime import datetime
from tqdm import tqdm, trange
from multiprocessing import Pool
import json

DATASET = 'Large'
PROJECT_PATH = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
DATA_PATH = osp.join(PROJECT_PATH, 'data', DATASET)
print("Data path:", DATA_PATH)

CONFIG = json.load(open('datagen.json', 'r'))
NAME = CONFIG['name']
SEED = CONFIG['seed']

def split_train_test(
    src_path: str = osp.join(DATA_PATH, 'KBQAP', 'raw'),
    tar_path: str = osp.join(DATA_PATH, 'KBQAP'),
    test_frac: float = 0.2,
):
    assert 0 < test_frac < 1
    assert osp.exists(src_path)
    os.makedirs(tar_path, exist_ok=True)
    path_train = osp.join(tar_path, 'train')
    path_test = osp.join(tar_path, 'test')
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)
    print(f'Splitting data from {src_path} to {path_train} and {path_test}...')
    
    file_list = os.listdir(src_path)
    random.shuffle(file_list)
    n_test = round(len(file_list) * test_frac)
    
    for i, file in enumerate(file_list):
        if i < n_test: os.rename(osp.join(src_path, file), osp.join(tar_path, 'test', file))
        else: os.rename(osp.join(src_path, file), osp.join(tar_path, 'train', file))
    os.rmdir(src_path)
            
def gen_synKBQAP(
    save_path: str = osp.join(DATA_PATH, 'KBQAP', 'raw'),
    n_graphs: int = 200,
    n1_scale = (5, 100),
    n2_mul_scale = (1.0, 2.5),
    sim_scale = (-1.0, 1.0),
    seed: int = 42
):
    os.makedirs(save_path, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    
    num_workers = 16
    
    with Pool(processes=num_workers) as pool:
        args = [(i, n1_scale, n2_mul_scale, sim_scale, save_path) for i in range(n_graphs)]
        list(tqdm(pool.starmap(process_task, args), total=n_graphs))
            
def process_task(i, n1_scale, n2_mul_scale, sim_scale, save_path):
    n1 = random.randint(n1_scale[0], n1_scale[1])
    n2_mul = random.uniform(n2_mul_scale[0], n2_mul_scale[1])
    n2 = int(n1 * n2_mul)

    result, idx = None, 0
    while result is None:
        idx += 1
        result, F1, F2, Kp = gen_problem(n1, n2, sim_scale)

    result_dict = KBQAP_to_dict(F1, F2, Kp, result)

    with open(os.path.join(save_path, f'{i:06d}.json'), 'w') as f:
        json.dump(result_dict, f)

    return i
            
def gen_problem(
    n1: int, 
    n2: int, 
    sim_scale = (-1.0, 1.0),
) -> dict:
    F1 = gen_symmetric_matrix(n1, value_scale=sim_scale, zero_diag=True)
    F2 = gen_symmetric_matrix(n2, value_scale=sim_scale, zero_diag=True)
    Kp = np.random.uniform(sim_scale[0], sim_scale[1], size=(n1, n2))
    
    result = {}
    X = gen_random_permute_matrix(n1, n2)
    result["X"] = X
    result["obj"] = np.sum(np.multiply(X, np.matmul(F1, np.matmul(X, F2)))) + np.sum(np.multiply(X, Kp))
    result["gap"] = -1
    result["opt"] = 0
    return result, F1, F2, Kp
            
def gen_symmetric_matrix(
    n: int, 
    value_scale = (0.0, 1.0),
    zero_diag: bool = True,
) -> np.ndarray:
    up_tri = np.random.uniform(value_scale[0], value_scale[1], size=(n, n))
    fg_mat = np.random.choice([0, 1], size=(n, n), p=[0, 1])
    up_tri = np.multiply(up_tri, fg_mat)
    
    up_tri = np.triu(up_tri, k=0)
    lo_tri = np.array(up_tri.T)
    np.fill_diagonal(lo_tri, 0.0)
    sym = up_tri + lo_tri
    
    if zero_diag: np.fill_diagonal(sym, 0)
    
    return sym

def gen_random_permute_matrix(n1, n2):
    assert n1 <= n2
    ids = np.random.choice(np.arange(n2), n1, replace=False)
    x = np.zeros((n1, n2), dtype=np.float32)
    x[np.arange(n1), ids] = 1
    return x

def KBQAP_to_dict(F1: np.ndarray, F2: np.ndarray, Kp: np.ndarray, result: dict) -> dict:
    dict = {
        "n1": F1.shape[0],
        "n2": F2.shape[0],
        "F1": F1.tolist(),
        "F2": F2.tolist(),
        "Kp": Kp.tolist(),
        "gt_X": result["X"].tolist(),
        "gt_obj": result["obj"]
    }
    return dict
    
def gen_synKBQAP_cfg(name, config):
    os.system(f"rm -rf {osp.join(DATA_PATH, name)}")
    
    gen_synKBQAP(
        save_path=osp.join(DATA_PATH, name, 'raw'),
        n_graphs=config['n_graphs'], 
        n1_scale=config['n1_scale'],
        n2_mul_scale=config['n2_mul_scale'],
        sim_scale=config['sim_scale'],
        seed=SEED,
    )
    split_train_test(
        src_path=osp.join(DATA_PATH, name, 'raw'),
        tar_path=osp.join(DATA_PATH, name),
        test_frac=config['test_frac']
    )

if __name__ == '__main__':
    
    for name in NAME:
        config = CONFIG['dataset'][name]
        gen_synKBQAP_cfg(name, config)