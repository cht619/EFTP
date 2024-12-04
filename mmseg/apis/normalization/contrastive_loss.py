import torch
import torch.nn.functional as F
from ..baselines.sourceonly import SourceOnly
from .ema import create_ema_model
from ..baselines.prompt_utils import position_by_uncertainty
from .loss_utils import construct_data_pair, cal_contrastive_loss_normal
from mmseg.models.utils import resize


class CL_base(SourceOnly):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lrs = kwargs.pop('lrs', [1e-6, 1e-5])
        self.model_ema = create_ema_model(self.model)

    def get_pl_from_source(self, data):
        # 直接用source model得到pseud-labels
        # 这里的resize最后是和输入一致，但是如果直接用tensor的话也会对不上，手动调整一下
        # 修改：直接按resize后的就行了
        with torch.no_grad():
            outputs = self.model_ema.test_step(data, mode='predict')
            pl = outputs[0].pred_sem_seg.data
            # seg_logits = resize(
            #     input=logits,
            #     size=data['inputs'][0].shape[1:],
            #     mode='bilinear',
            #     align_corners=False)
            # pl = seg_logits.softmax(1).argmax(1).long()
            # pl.shape=[1,H,W] if bs =1
        return pl.squeeze()

    def get_optimizer(self):
        # self.model.requires_grad_(False)

        for np, p in self.model.named_parameters():
            p.requires_grad = True

        params_list = [
            {'params': self.model.backbone.parameters(), 'lr': 1e-6},
            {'params': self.model.decode_head.parameters(), 'lr': 1e-6},
        ]

        optimizer = torch.optim.Adam(params_list, betas=(0.9, 0.999))  # bs=1
        print('len params to train:{}. lr is {}'.format(len(params_list), self.lrs))
        return optimizer


class CL(CL_base):
    def forward_loss(self, data, pl, optimizer):
        optimizer.zero_grad()
        outputs = self.model.test_step(data)
        seg_logits = outputs[0].seg_logits.data
        probs = seg_logits.softmax(0).max(0)[0].flatten()
        seg_logits = seg_logits.permute(1, 2, 0).reshape(-1, 19)  # []

        # get class prototype
        # class_prototypes = torch.tensor([]).cuda()
        class_prototypes = {}
        with torch.no_grad():
            for i in range(19):
                mask = (pl == i).flatten()
                if mask.sum() > 0:
                    seg_logits_c = seg_logits[mask, :]
                    # class_prototypes = torch.cat((class_prototypes, seg_logits_c.mean(0, keepdim=True)), 0)
                    class_prototypes[i] = seg_logits_c.mean(0, keepdim=True).detach()

        contr_loss = 0
        for i in range(19):
            mask = (pl == i).flatten()
            if mask.sum() > 0:
                optimizer.zero_grad()
                seg_logits = self.model.test_step(data)[0].seg_logits.data.permute(1, 2, 0).reshape(-1, 19)
                seg_logits_c = seg_logits[mask, :]  # [n, 19]
                probs_c = probs[mask]
                probs_sort_i = probs_c.argsort()  # default is ascending order
                # probs_c_stable = probs_c[probs_sort_i[:100]]  # stable
                # probs_c_unstable = probs_c[probs_sort_i[-100:]]  # unstable

                fea_neg = torch.cat([class_prototypes[j] for j in class_prototypes.keys() if i != j])
                fea_pos = torch.cat((seg_logits_c[probs_sort_i[:100], :], class_prototypes[i]))
                labels = torch.ones(fea_pos.shape[0] + fea_neg.shape[0]).cuda().detach()
                labels[-len(fea_pos):] = 0
                labels = labels.unsqueeze(1).detach()
                feas = torch.cat((fea_pos, fea_neg), 0)
                feas = F.normalize(feas, p=2, dim=1)

                logits = torch.div(torch.matmul(feas, feas.T), 1.0)
                logits_max, _ = torch.max(logits, dim=1, keepdim=True)
                logits = logits - logits_max.detach()
                mask = torch.eq(labels, labels.T).float().detach()
                logits_mask = torch.scatter(
                    torch.ones_like(mask), 1, torch.arange(feas.shape[0]).view(-1, 1).cuda(), 0)  # 这里就是把对角线设为0
                mask = mask * logits_mask  # [n, n]
                exp_logits = torch.exp(logits) * logits_mask  # [n, n]
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # sum1就是把每个样本与其他样本的距离累加起来，这里记得是已经包含了负样本的距离了
                mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)  # 所以后面直接算正样本就没问题的
                loss = -1 * mean_log_prob_pos.mean()
                loss.backward(retain_graph=True)
                optimizer.step()

    def forward_epoch(self, epoch, data, optimizer=None):
        gt = data['data_samples'][0].gt_sem_seg.data.long().cuda().detach()

        with torch.no_grad():
            outputs = self.model.test_step(data)

        self.forward_loss(data, gt, optimizer)
        return outputs


class CL_TS(CL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_ema = create_ema_model(self.model)
        self.alpha_teacher = 0.99

    @torch.no_grad()
    def ema_update(self, epoch=None):
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data[:] = self.alpha_teacher * ema_param[:].data[:] + (1 - self.alpha_teacher) * param[:].data[:]

    def forward_epoch(self, epoch, data, optimizer=None):
        gt = data['data_samples'][0].gt_sem_seg.data.long().cuda().detach()

        with torch.no_grad():
            outputs = self.model_ema.test_step(data)

        self.forward_loss(data, gt, optimizer)
        self.ema_update()
        return outputs

    
class CL_Tuning(CL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.select_samples = 0.1  # 每个类的10%有多少数目？

    def forward_loss(self, data, pl, optimizer):
        optimizer.zero_grad()
        feas = self.model.extract_feat(data['inputs'][0].cuda().float().unsqueeze(0))
        feas = self.model.decode_head.forward(feas, return_feas=True)  # [bs,256, 135, 240]
        logits = resize(
            input=feas, size=(1080, 1920), mode='bilinear', align_corners=False).squeeze()
        pl = self.get_pl_from_source(data)  # [H, W]
        pl_flatten = pl.flatten()
        # outputs = self.model.test_step(data)
        # logits = outputs[0].seg_logits.data
        logits_flatten = torch.permute(logits, (1, 2, 0)).flatten(0, 1)
        uncertainty = position_by_uncertainty(model=self.model, n_prompts=2000, data=data, return_uncertainty=True)
        uncertainty_flatten = torch.flatten(uncertainty)  # uncertainty.shape=[H, W]
        feas, labels = construct_data_pair(pl_flatten, logits_flatten, uncertainty_flatten)
        loss_contrastive = cal_contrastive_loss_normal(feas, labels)
        loss_contrastive.backward()
        optimizer.step()

    def forward_epoch(self, epoch, data, optimizer=None):
        gt = data['data_samples'][0].gt_sem_seg.data.long().cuda().detach()

        with torch.no_grad():
            outputs = self.model.test_step(data)

        self.forward_loss(data, gt, optimizer)
        return outputs


