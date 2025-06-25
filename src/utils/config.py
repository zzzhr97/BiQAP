import yaml
import os
import os.path as osp
from datetime import datetime

from src.utils.path import get_project_path
from src.utils.logger import setup_logger, INFO, DEBUG


PROJECT_PATH = get_project_path(3)
RESULT_PATH = osp.join(PROJECT_PATH, 'experiments')

class Config(object):
    def __init__(self, cfg_path):
        self.yaml_path = osp.join(PROJECT_PATH, cfg_path)
        with open(self.yaml_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)
            
        dname = self.cfg["dataset"]["name"]
        if type(dname) == list: dname = "+".join(dname)
        elif type(dname) == int: dname = str(dname)
            
        # archive
        os.makedirs(RESULT_PATH, exist_ok=True)
        for datadir in os.listdir(RESULT_PATH):
            if dname not in datadir or osp.isfile(osp.join(RESULT_PATH, datadir)): continue
            for expdir in os.listdir(osp.join(RESULT_PATH, datadir)):
                if not expdir.endswith('complete') and not expdir.endswith('archive'):
                    ll = expdir.split("_")[:-3]
                    if ll[0] != self.cfg['mode']: continue
                    if "_".join(ll) != self.cfg['name']: continue
                    
                    fulldir = osp.join(RESULT_PATH, datadir, expdir)
                    os.system(f'mv {fulldir} {fulldir}_archive')
                    INFO(f'Archived: {fulldir}')
            
        # result dir
        time_id = datetime.now().strftime('_%Y%m%d_%H%M%S_%f')
        self.result_dir = osp.join(RESULT_PATH, dname,
            self.cfg["name"] + time_id)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # logger
        logpath = osp.join(self.result_dir, 'result.log')
        setup_logger(logpath)
        
        # model save
        self.model_path = osp.join(self.result_dir, 'model')
        
        # copy yaml
        os.system(f'cp {self.yaml_path} {self.result_dir}/')
        
        INFO(f"name: {self.cfg['name']}, mode: {self.cfg['mode']}")
        
    def get_model_dir(self):
        return self.model_path
    
    def get_cfg(self):
        return self.cfg
    
    def get_result_dir(self):
        return self.result_dir