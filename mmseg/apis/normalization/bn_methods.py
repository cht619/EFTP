import torch
from torch import nn
from tqdm import tqdm
from mmengine.runner import autocast
from ..baselines.sourceonly import SourceOnly
from mmseg.models.utils import resize


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    # x should be [b, c, h, w]
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class TENT(SourceOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lr = kwargs.pop('lr', 1e-4)

    def get_optimizer(self):
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

    def forward_epoch(self, epoch, data, optimizer=None):
        optimizer.zero_grad()
        with autocast(enabled=self.fp16):
            outputs = self.model.test_step(data)  # bs =1
            logits = self.model.test_step(data, mode='tensor')  # [bs, c, h, w]
            logits = resize(logits, size=data['data_samples'][0].metainfo['ori_shape'], mode='bilinear',
                            align_corners=False, warning=False)
            loss_train = softmax_entropy(logits).mean()
            loss_train.backward()
            optimizer.step()
        return outputs

    def forward_train(self):
        optimizer = self.get_optimizer()
        tbar = tqdm(self.dataloader)
        for epoch, data in enumerate(tbar):
            self.set_model_status()
            outputs = self.forward_epoch(epoch, data, optimizer)

            if self.efficient_test:
                self.save_result(outputs.pred_sem_seg.data, outputs.img_path.split('/')[-1][:-4])
            else:
                self.evaluator.process(data_samples=outputs, data_batch=data)

        if not self.efficient_test:
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            print(metrics)
