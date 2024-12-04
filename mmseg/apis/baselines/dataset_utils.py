from copy import deepcopy
from mmengine.dataset import ConcatDataset, default_collate
from torch.utils.data import DataLoader

# model merging continual TTA
def get_continual_dataloaders(cfg, runner, domains):
    train_dataloader_cfg = deepcopy(cfg.test_dataloader)
    train_dataloader_cfg.sampler.shuffle = False  # test

    dataloaders_dict = {}  # multi domains
    for domain in domains:
        if domain in ['fog', 'snow', 'rain', 'night']:
            train_dataloader_cfg.dataset.data_prefix.img_path = 'rgb_anon/{}/train'.format(domain)
            train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gt/{}/train'.format(domain)
        elif domain in ['Rio', 'Rome', 'Taipei', 'Tokyo']:
            train_dataloader_cfg.dataset.data_prefix.img_path = '{}/Images/Test'.format(domain)
            train_dataloader_cfg.dataset.data_prefix.seg_map_path = '{}/Labels/Test'.format(domain)

        train_dataloader_cfg.dataset.pipeline = cfg.test_pipeline
        train_dataloader = runner.build_dataloader(dataloader=train_dataloader_cfg)
        dataloaders_dict[domain] = train_dataloader

    return dataloaders_dict


# build dataset for adaptation
def get_cs_foggy_rainy_dataloaders(cfg, runner, domain='csfoggy'):
    train_dataloader_cfg = deepcopy(cfg.test_dataloader)
    train_dataloader_cfg.sampler.shuffle = False  # test
    train_dataloader_cfg.dataset.data_root = './data/Segmentation/Cityscapes'
    if domain == 'csfoggy':
        train_dataloader_cfg.dataset.type = 'CityscapesDataset_foggy'
        train_dataloader_cfg.dataset.data_prefix.img_path = 'leftImg8bit_foggyDBF/val'
        train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gtFine/val'
        train_dataloader_cfg.dataset.train_txt = './mmseg/datasets/data_information/train_cs_foggy.txt'
    elif domain == 'csrainy':
        train_dataloader_cfg.dataset.type = 'CityscapesDataset_rainy'
        train_dataloader_cfg.dataset.data_prefix.img_path = 'leftImg8bit_rain/val'
        train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gtFine/val'
        train_dataloader_cfg.dataset.train_txt = './mmseg/datasets/data_information/train_cs_rainy.txt'
    else:
        return None

    train_dataloader_cfg.dataset.pipeline = cfg.test_pipeline
    train_dataloader = runner.build_dataloader(dataloader=train_dataloader_cfg)
    return train_dataloader


def build_concat_dataset(cfg, runner, domain=None):
    train_dataloader_cfg = deepcopy(cfg.test_dataloader)
    domains = ['fog', 'snow', 'rain', 'night']
    concat_datasets = []
    train_dataloader_cfg.dataset.type = 'ACDCDataset'
    train_dataloader_cfg.dataset.data_root = './data/Segmentation/ACDC'
    for domain in domains:
        train_dataloader_cfg.dataset.data_prefix.img_path = 'rgb_anon/{}/train'.format(domain)
        train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gt/{}/train'.format(domain)

        train_dataloader_cfg.dataset.pipeline = cfg.test_pipeline
        train_dataloader = runner.build_dataloader(dataloader=train_dataloader_cfg)
        concat_datasets.append(train_dataloader.dataset)

    concat_dataloader = DataLoader(ConcatDataset(
        datasets=concat_datasets), batch_size=1, num_workers=16, shuffle=False, collate_fn=default_collate)
    return concat_dataloader