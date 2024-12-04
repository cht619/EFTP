from mmseg.apis.baselines.fl_seg_baselines import ModelMerging


def refine_baseline(cfg):
    cfg.checkpoint = './checkpoints/FedSeg/{}/model.pth'.format(str.lower(cfg.domain))
    if any(s in cfg.baseline for s in ['FedSeg_TTA', 'ModelMerging']):
        cfg.model.backbone.type = 'mit_b5_daformer_EVP'  # 为了实现model merging，需要用旧的
        cfg.checkpoint = './checkpoints/FedSeg_Prompt/{}/model_old.pth'.format(
            str.lower(cfg.domain))

    return cfg


def refine_dataset(cfg):
    if cfg.test_dataloader.dataset.type == 'ACDCDataset':
        domain = cfg.test_dataloader.dataset.data_prefix.img_path.split('/')[-2]
        cfg.test_dataloader.dataset.data_prefix.img_path = (
            cfg.test_dataloader.dataset.data_prefix.img_path.replace(domain, cfg.domain))
        cfg.test_dataloader.dataset.data_prefix.seg_map_path = (
            cfg.test_dataloader.dataset.data_prefix.seg_map_path.replace(domain, cfg.domain))

    if cfg.test_dataloader.dataset.type == 'NTHUDataset':
        cfg.test_dataloader.dataset.data_prefix.img_path = '{}/Images/Test'.format(cfg.domain)
        cfg.test_dataloader.dataset.data_prefix.seg_map_path = '{}/Labels/Test'.format(cfg.domain)

    return cfg


def refine_cfg(cfg):
    cfg = refine_baseline(cfg)
    cfg = refine_dataset(cfg)
    return cfg


def build_solver(baseline):
    print('baseline is:', baseline)
    solver_dicts = {
        'ModelMerging': ModelMerging
        }
    return solver_dicts[baseline]