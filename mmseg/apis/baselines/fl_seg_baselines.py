import torch
from torch import nn, optim
import torch.nn.functional as F
from torchsummary import summary
import os
import csv
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from mmseg.models.utils import resize
from torch.utils.data import DataLoader, Dataset
from mmengine.dataset import default_collate, ConcatDataset
from .utils import (get_named_submodule, set_named_submodule, MoE, WMoE,
                    get_continual_dataloaders)
from .loss_utils import softmax_entropy
from .sourceonly import SourceOnly


def create_global_model(model):
    global_model = deepcopy(model)  #get_model(args.model)(num_classes=num_classes)

    for param in global_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(global_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #global_model = torch.nn.DataParallel(global_model, device_ids=availble_gpus)
    return global_model


class EFTP(SourceOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = kwargs['cfg']
        self.runner = kwargs['runner']
        self.domain = kwargs['domain']
        if kwargs['domain'] in ['fog', 'snow', 'rain', 'night']:
            self.domains = ['fog', 'night', 'rain', 'snow']
        elif kwargs['domain'] in ['Rio', 'Rome', 'Taipei', 'Tokyo']:
            self.domains = ['Rio', 'Rome', 'Taipei', 'Tokyo']
        else:
            self.domains = ['cs']
        self.model_global = create_global_model(self.model)
        self.upscaling_layer()
        self.model_initialization()

    def upscaling_layer(self, tqdm_desc="Upscaling Linear Modules", replaced_layer=(nn.Linear, ),
                        select_linears=['attn.proj']):
        # 这里直接先添加gate。对model_ema改进
        # select_linears 要不要直接对prompt的linear做改进
        base_backbone = deepcopy(self.model_global.backbone)

        upscaling_names = []
        for name, module in tqdm(
                tuple(self.model_global.backbone.named_modules()),
                tqdm_desc,
                leave=True,
                dynamic_ncols=True,
        ):
            if isinstance(module, replaced_layer):
                if any(n in name for n in select_linears):  # only choose some linear layers
                    linear_pretrained = deepcopy(get_named_submodule(base_backbone, name))
                    weightEnsemblingLinear = WMoE(
                        base_model=linear_pretrained,
                        init_lambda=0.3,
                        batch_first=True,
                        router_hidden_layers=2,
                        batch_reduce=True,
                    ).cuda()
                    set_named_submodule(base_backbone, name, weightEnsemblingLinear)

                    upscaling_names.append(name)

        self.upscaling_names = upscaling_names  # save upscaling_names
        print('Replaced {} layers to WMoE'.format(len(upscaling_names)))
        self.model_global.backbone = base_backbone

    def get_dataloader(self):
        dataloaders = []
        datasets = []
        if self.domains[0] != 'cs':  # cityscapes
            self.domains.remove(self.domain)  # remaining domains as training data
        self.cfg.test_dataloader.sampler.shuffle = True
        for d in self.domains:
            print('load domain:', d)
            train_dataloader_cfg = deepcopy(self.cfg.test_dataloader)
            if self.domain in ['fog', 'snow', 'rain', 'night']:
                train_dataloader_cfg.dataset.data_prefix.img_path = 'rgb_anon/{}/train'.format(d)
                train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gt/{}/train'.format(d)
            elif self.domain in ['Rio', 'Rome', 'Taipei', 'Tokyo']:
                train_dataloader_cfg.dataset.data_prefix.img_path = '{}/Images/Test'.format(d)
                train_dataloader_cfg.dataset.data_prefix.seg_map_path = '{}/Labels/Test'.format(d)
            else:  # cityscapes
                train_dataloader_cfg.dataset.data_prefix.img_path = 'leftImg8bit/train'
                train_dataloader_cfg.dataset.data_prefix.seg_map_path = 'gtFine/train'

            train_dataloader_cfg.dataset.pipeline = self.cfg.test_pipeline
            train_dataloader = self.runner.build_dataloader(dataloader=train_dataloader_cfg)
            dataloaders.append(train_dataloader)
            datasets.append(train_dataloader.dataset)

        train_dataloader = DataLoader(ConcatDataset(datasets), batch_size=4, num_workers=8, shuffle=True,
                                      collate_fn=default_collate)

        return train_dataloader

    # initialize: prompt and router
    def model_initialization(self):
        local_model = deepcopy(self.model)

        ema_model_params = []
        for n, p in self.model_global.named_parameters():
            if any(s in n for s in ['prompt', 'gate', 'attn.proj']):  # update
                p.requires_grad = True
                ema_model_params.append(p)
            else:
                p.requires_grad = False

        local_model_params = []
        for n, p in local_model.named_parameters():
            local_model_params.append(p)
            p.requires_grad = True
        all_params_list = [
            {'params': ema_model_params, 'lr': 1e-4},  # local model prompt
            {'params': local_model_params, 'lr': 1e-4},  # expert gate
        ]

        optimizer = optim.AdamW(all_params_list, lr=1e-5, weight_decay=0.01)
        self.model_global.eval()
        train_dataloader = self.get_dataloader()
        for data in tqdm(train_dataloader):
            data = self.process_batch_class_label_nthu(data)
            optimizer.zero_grad()
            batch_size = len(data['inputs'])
            # print(data['data_samples'][0].img_path, data['data_samples'][0].seg_map_path)
            gts = torch.cat([data['data_samples'][i].gt_sem_seg.data.long() for i in range(batch_size)]).cuda()
            outputs = self.model_global.test_step(data)
            seg_logits = torch.cat([outputs[i].seg_logits.data.unsqueeze(0) for i in range(batch_size)]).cuda()
            loss_train = self.model_global.decode_head.loss_decode(seg_logits, gts, ignore_index=255)
            if loss_train < 1.2:  # threshold \tau
                loss_train.backward()
                optimizer.step()

                for name, module in self.model_global.backbone.named_modules():
                    if isinstance(module, WMoE):
                        if name in self.upscaling_names:
                            linear_expert = deepcopy(get_named_submodule(local_model.backbone, name))
                            module.update_expert_model(linear_expert.state_dict())

    # consistency loss
    def forward_loss(self, data, optimizer):
        outputs_logits_ema = self.model_global.test_step(data)[0].seg_logits.data.unsqueeze(0)
        outputs_logits = self.model.test_step(data)[0].seg_logits.data.unsqueeze(0)

        prob_a = F.log_softmax(outputs_logits.reshape(-1, 19), dim=1)
        prob_b = F.softmax(outputs_logits_ema.reshape(-1, 19), dim=1)

        kl_loss = F.kl_div(prob_a, prob_b, reduction='batchmean')
        kl_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # test-time model merging
    def model_merging(self, data):
        for name, module in self.model_global.backbone.named_modules():
            if isinstance(module, WMoE):
                if name in self.upscaling_names:
                    linear_expert = deepcopy(get_named_submodule(self.model.backbone, name))
                    module.update_expert_model(linear_expert.state_dict())

        with torch.no_grad():
            outputs = self.model_global.test_step(data)

        return outputs

    # get result
    def forward_epoch(self, epoch, data, optimizer=None):
        optimizer.zero_grad()
        self.forward_loss(data, optimizer)  # update local model
        outputs = self.model_merging(data)
        return outputs

    def forward_train(self):
        optimizer = self.get_optimizer()

        with tqdm(total=len(self.dataloader)) as pbar:
            pbar.set_description('--Current dataset is {}'.format('ACDC all'))
            for epoch, data in enumerate(self.dataloader):
                # data = self.process_batch_class_label_nthu(data)  # preprocess nthu data
                self.set_model_status()
                outputs = self.forward_epoch(epoch, data, optimizer=optimizer)

                self.evaluator.process(data_samples=outputs, data_batch=data)
                pbar.update(1)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        each_class_result = self.evaluator.metrics[0].each_class_result  # .metrics[0]
        with open(os.path.join(self.work_dir, 'result.csv'), 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(['ACDC ALL', metrics['aAcc'], metrics['mAcc'], metrics['mIoU']])

            writer.writerow(each_class_result['Class'])
            writer.writerow(each_class_result['IoU'])
        torch.save(metrics, os.path.join(self.work_dir, 'result.pkl'))  # if need

