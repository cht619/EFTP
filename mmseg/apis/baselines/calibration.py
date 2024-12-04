import torch
from torch import nn, optim
import os
import csv
from tqdm import tqdm
from ..baselines import SourceOnly
from mmseg.models.utils import resize
from mmengine.structures import PixelData


class CalibrationExP(SourceOnly):
    def __init__(self, **kwargs):
        super(CalibrationExP, self).__init__(**kwargs)
        # shape should be like: [bs, H, W]
        self.gts, self.preds, self.probs = torch.tensor([]), torch.tensor([]), torch.tensor([])
        self.lr = 1e-5

    def data2list(self, data):
        # change format
        num_augs = len(data[next(iter(data))])  # e.g., 14
        # 把每一个x_aug存到list里面。  [{inputs:xxx, data_samples:}, dict, ..., dict_{n_aug}]
        data_list = [{key: value[idx]
                      for key, value in data.items()}
                     for idx in range(num_augs)]  # len=num_augs
        return data_list

    def merge_preds(self, data_samples_list):
        predictions = []
        seg_preds = torch.tensor([]).cuda()
        for data_samples in data_samples_list:  # bs level
            seg_pred_aug = torch.tensor([]).cuda()
            seg_logits = data_samples[0].seg_logits.data
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:  # num_augs level
                seg_logit = data_sample.seg_logits.data
                if self.model.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
                seg_pred_aug = torch.cat(
                    (seg_logit.detach().unsqueeze(0), seg_pred_aug), 0)
            logits /= len(data_samples)  # mean output
            if self.model.out_channels == 1:
                seg_pred = (logits > self.model.decode_head.threshold).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})  # 这里直接取最后一个了
            if hasattr(data_samples[0], 'gt_sem_seg'):
                data_sample.set_data(
                    {'gt_sem_seg': data_samples[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': data_samples[0].img_path})
            predictions.append(data_sample)  # length = bs
            seg_preds = torch.cat((seg_pred_aug.unsqueeze(0), seg_preds), 0)
        # print(torch.tensor(seg_preds[0]).shape, len(seg_preds))  # 这里不能直接转tensor，估计得用concat
        return predictions, seg_preds.detach()  # [bs, n_augs, c, h, w]

    def get_optimizer(self, mode='tent'):
        if mode == 'tent':
            self.model.train()
            self.model.requires_grad_(False)
            param_list = []
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for np, p in m.named_parameters():  # 5*2=10
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            p.requires_grad = True
                            param_list.append(p)

            optimizer = torch.optim.Adam(param_list, lr=self.lr, betas=(0.9, 0.999))  # bs=1
            print('len params to train:{}. lr is {}'.format(len(param_list), self.lr))
            return optimizer

        elif mode == 'CoTTA':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))  # bs=1
            return optimizer

        elif mode == 'prompt':
            self.model.train()
            self.model.requires_grad_(False)
            param_list = []
            for n, p in self.model.named_parameters():
                # note: classifier是'decode_head'
                if any(s in n for s in ['prompt']):
                    p.requires_grad = True
                    param_list.append(p)
            optimizer = torch.optim.Adam(param_list, lr=self.lr, betas=(0.9, 0.999))  # bs=1
            print('len params to train:{}. lr is {}'.format(len(param_list), self.lr))
            return optimizer

    def adapt_test(self, data, optimizer, mode='source'):
        if mode == 'source':
            with torch.no_grad():
                outputs = self.model.test_step(data)
        elif mode == 'tent':
            outputs = self.model.test_step(data)  # bs =1
            logits = outputs[0].seg_logits.data  # [bs, c, h, w]
            loss_train = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
            loss_train.backward()
            optimizer.step()

        elif mode == 'cotta':
            data_list = self.data2list(data)
            num_augs = len(data_list)
            for i in range(num_augs):
                data_list[i] = self.process_batch_class_label_nthu(data_list[i])
            predictions = []
            with torch.no_grad():
                for data_ in data_list:
                    predictions.append(self.model.test_step(data_))  # n_augs SegDataSample
                outputs, seg_preds = self.merge_preds(list(zip(*predictions)))

            return outputs, seg_preds.squeeze(0).cpu().detach()  # [6, 19, 1080, 1920]

        elif mode == 'prompt':
            outputs = self.model.test_step(data)

        return outputs

    def forward_epoch(self, epoch, data, optimizer=None, mode='prompt'):
        batch_size = len(data['inputs'])

        if mode == 'cotta':
            outputs, seg_preds = self.adapt_test(data, optimizer, mode=mode)
            data_list = self.data2list(data)
            self.gts = torch.cat((self.gts, data_list[3]['data_samples'][0].gt_sem_seg.data.long().cpu().unsqueeze(0)), 0)
            probs, preds = torch.max(torch.softmax(seg_preds, 1).mean(0), 0)
        else:
            outputs = self.adapt_test(data, optimizer, mode=mode)
            seg_logits = outputs[0].seg_logits.data.cpu().detach()
            gts = torch.cat([data['data_samples'][i].gt_sem_seg.data.long() for i in range(batch_size)]).cuda()
            self.gts = torch.cat((self.gts, gts[0].cpu().unsqueeze(0)), 0)
            probs, preds = torch.max(torch.softmax(seg_logits, 0), 0)

        self.probs = torch.cat((self.probs, probs.unsqueeze(0)), 0)
        self.preds = torch.cat((self.preds, preds.unsqueeze(0)), 0)
        return outputs

    def forward_train(self, mode='prompt'):
        optimizer = self.get_optimizer(mode=mode)
        tbar = tqdm(self.dataloader)
        # tbar = self.dataloader
        for epoch, data in enumerate(tbar):
            # data = self.process_batch_class_label_nthu(data)
            self.set_model_status()
            outputs = self.forward_epoch(epoch, data, optimizer=optimizer, mode=mode)
            self.evaluator.process(data_samples=outputs, data_batch=data)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        print(metrics)
        with open(os.path.join(self.work_dir, 'result.csv'), 'a+') as f:
            writer = csv.writer(f)
            writer.writerow([metrics['aAcc'], metrics['mAcc'], metrics['mIoU']])
        torch.save(metrics, os.path.join(self.work_dir, 'result.pkl'))

        state = {'gts': self.gts, 'probs': self.probs, 'preds': self.preds}
        torch.save(state, '{}/analysis.pkl'.format(self.work_dir))



