import os
import sys  # 记得每次都要把路径加上，不要import同名的module有问题
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
current_path = os.path.dirname(__file__)
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner, load_checkpoint
from tqdm import tqdm



def run_computation_time(cfg_file='configs/tta/train_cs_.py', domain='cs'):
    checkpoint = './checkpoints/FedSeg_Prompt/{}/model.pth'.format(domain)
    cfg = Config.fromfile(cfg_file)
    cfg.work_dir = './Run/ComputationTIme'
    # change domain
    runner = Runner.from_cfg(cfg)
    model = runner.model
    dataloader = runner.test_dataloader
    tbar = tqdm(dataloader)
    with torch.no_grad():
        for data in tbar:
            model.test_step(data)



if __name__ == '__main__':
    run_computation_time()
