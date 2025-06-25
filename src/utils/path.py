import os
import os.path as osp

def get_project_path(depth: int):
    return osp.abspath(osp.join(__file__, *([osp.pardir] * depth)))