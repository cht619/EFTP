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


def dice_score(pred, truth, num_classes=None):
    """
    Calculate Dice Score for a single class or multiple classes.

    Parameters:
    - pred: numpy array, the predicted segmentation (H x W for single class, H x W x C for multi-class).
    - truth: numpy array, the ground truth segmentation (same shape as pred).
    - num_classes: int, number of classes (for multi-class). If None, assumes binary segmentation.

    Returns:
    - dice: float for single class, or list of floats for multi-class.
    """
    pred = np.asarray(pred)
    truth = np.asarray(truth)

    if num_classes is None:  # Binary segmentation
        intersection = np.sum((pred == 1) & (truth == 1))
        pred_sum = np.sum(pred == 1)
        truth_sum = np.sum(truth == 1)
        dice = (2 * intersection) / (pred_sum + truth_sum + 1e-6)  # Add epsilon to avoid division by zero
        return dice
    else:  # Multi-class segmentation
        dice_scores = []
        for c in range(num_classes):
            intersection = np.sum((pred == c) & (truth == c))
            pred_sum = np.sum(pred == c)
            truth_sum = np.sum(truth == c)
            dice = (2 * intersection) / (pred_sum + truth_sum + 1e-6)  # Add epsilon to avoid division by zero
            dice_scores.append(dice)
        return dice_scores


# 前置准备
def run_prepare(cfg_file='configs/tta/train_acdc.py', domain='fog'):
    print(domain)
    domain_jpg_dict = {
        'fog': 'GOPR0478_frame_000195_rgb_anon.png',
        'night': 'GOPR0376_frame_000954_rgb_anon.png',
        'rain': 'GOPR0402_frame_000605_rgb_anon.png',
        'snow': 'GOPR0606_frame_000503_rgb_anon.png',
    }
    # 如果需要model merging才需要model_old.pth 这里我只需要看看source only的结果如何先
    checkpoint = './checkpoints/FedSeg_Prompt/{}/model.pth'.format(domain)
    cfg = Config.fromfile(cfg_file)
    cfg.work_dir = './Run/Visualization'
    cfg.model.backbone.type = 'MiT_EVP'  # MiT_EVP是新的MiT加上EVP的实现
    # change domain
    domain_ = cfg.test_dataloader.dataset.data_prefix.img_path.split('/')[-2]
    cfg.test_dataloader.dataset.data_prefix.img_path = (
        cfg.test_dataloader.dataset.data_prefix.img_path.replace(domain_, domain))
    cfg.test_dataloader.dataset.data_prefix.seg_map_path = (
        cfg.test_dataloader.dataset.data_prefix.seg_map_path.replace(domain_, domain))
    runner = Runner.from_cfg(cfg)
    # build model
    model = runner.model
    model.load_state_dict(torch.load(checkpoint))
    # build dataloader
    dataloader = runner.test_dataloader
    for data in dataloader:
        file_name = data['data_samples'][0].img_path.split('/')[-1]
        if domain_jpg_dict[domain] in file_name:

            gt = data['data_samples'][0].gt_sem_seg.data
            output = model.test_step(data)
            pred = output[0].pred_sem_seg.data
            dice_score_ = dice_score(pred.squeeze(0).cpu().numpy(), gt.squeeze(0).cpu().numpy(), num_classes=19)
            return


if __name__ == '__main__':
    # 计算图片的dice_score
    run_prepare(domain='snow')
