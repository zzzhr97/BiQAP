import logging
import os
import os.path as osp
from datetime import datetime
from tqdm import tqdm

from src.utils.path import get_project_path


PROJECT_PATH = get_project_path(3)
    
def setup_logger(logpath):
    """This function is only used in config.py to setup the logger."""
    os.makedirs(osp.dirname(logpath), exist_ok=True)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(logpath)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
def INFO(msg):
    logger = logging.getLogger('main')
    logger.info(msg)
    tqdm.write(msg)
    return msg
    
def DEBUG(msg):
    logger = logging.getLogger('main')
    logger.debug(msg)
    tqdm.write(msg)
    return msg