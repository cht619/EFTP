import torch
from tqdm import tqdm
from mmengine.runner import autocast
from mmengine.structures import PixelData
from copy import deepcopy
from ..baselines.sourceonly import SourceOnly
from mmseg.models.utils import resize


def create_ema_model(model):
    ema_model = deepcopy(model)  #get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    return ema_model


class TeaStu_base(SourceOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tea_model = create_ema_model(self.model)
        self.img_id = 4  # 第4个刚好就是scale=1.0的大小，即原始大小
        self.lr = kwargs.pop('lr', 1e-4)
        self.alpha_teacher = kwargs.pop('alpha_teacher', 0.999)

    @torch.no_grad()
    def ema_update(self, epoch=None):
        for ema_param, param in zip(self.tea_model.parameters(), self.model.parameters()):
            ema_param.data[:] = self.alpha_teacher * ema_param[:].data[:] + (1 - self.alpha_teacher) * param[:].data[:]

    def set_model_status(self):
        self.model.eval()
        self.tea_model.eval()

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
        return predictions, seg_preds  # [bs, n_augs, h, w]

    def get_optimizer(self):
        param_list = []
        for np, p in self.model.named_parameters():
            if p.requires_grad:
                param_list.append(p)
        optimizer = torch.optim.Adam(param_list, lr=self.lr, betas=(0.9, 0.999))  # bs=1
        print('len params to train:{}. lr is {}'.format(len(param_list), self.lr))
        return optimizer

    def forward_loss(self, data_list, pl_list, optimizer):
        logits = self.model.test_step(data_list[self.img_id], mode='tensor')
        logits = resize(logits, size=data_list[self.img_id]['data_samples'][0].metainfo['ori_shape'], mode='bilinear',
                        align_corners=False, warning=False)
        loss_train = self.model.decode_head.loss_decode(
            logits, pl_list[:, self.img_id].long(), ignore_index=255)  # label is 3D tensor
        loss_train.backward()
        optimizer.step()

    def forward_epoch(self, epoch, data, optimizer=None):
        optimizer.zero_grad()
        # data.keys: ['inputs', 'data_samples']. 两个都是list，长度为bs
        num_augs = len(data[next(iter(data))])  # e.g., 14
        # 把每一个x_aug存到list里面。  [{inputs:xxx, data_samples:}, dict, ..., dict_{n_aug}]
        data_list = [{key: value[idx]
                      for key, value in data.items()}
                     for idx in range(num_augs)]  # len=num_augs
        predictions = []
        for data_ in data_list:
            with torch.no_grad():  # 不然爆显存
                with autocast(enabled=self.fp16):
                    predictions.append(self.tea_model.test_step(data_))  # n_augs SegDataSample
        # *就是把batch提出来，比如一共有n个aug，第一个是[x_0, ..., x_bs个]
        # 然后zip打包，这里明显就是按bs的维度来打包了，最终格式就有bs个，每个是[n_aug个]，
        # 比如2个bs2个aug就是[[x_aug0, x_aug1], [y_aug0, y_aug1]]
        outputs, seg_preds = self.merge_preds(list(zip(*predictions)))
        self.forward_loss(data_list, seg_preds, optimizer)

        # for np, p in self.model.named_parameters():
        #     print(p.data.sum())
        #     break

        return outputs

    def forward_train(self):
        optimizer = self.get_optimizer()
        tbar = tqdm(self.dataloader)
        for epoch, data in enumerate(tbar):
            self.set_model_status()
            outputs = self.forward_epoch(epoch, data, optimizer)
            self.ema_update()
            if self.efficient_test:
                self.save_result(outputs.pred_sem_seg.data, outputs.img_path.split('/')[-1][:-4])
            else:
                self.evaluator.process(data_samples=outputs, data_batch=data)

        if not self.efficient_test:
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            print(metrics)


