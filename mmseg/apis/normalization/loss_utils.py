import torch
import torch.nn.functional as F


def cal_contrastive_loss_normal(feas_norm, labels, temperature=1.0):
    # feas_norm.shape = [n, dim]. labels.shape=[n, 1]
    logits = torch.div(torch.matmul(feas_norm, feas_norm.T), 1.0)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    mask = torch.eq(labels, labels.T).float().detach()
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1, torch.arange(feas_norm.shape[0]).view(-1, 1).cuda(), 0)  # 这里就是把对角线设为0
    mask = mask * logits_mask  # [n, n]
    exp_logits = torch.exp(logits) * logits_mask  # [n, n]
    # sum1就是把每个样本与其他样本的距离累加起来，这里记得是已经包含了负样本的距离了
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # 原文公式中的Ai其实是包含了正样本，所以分母这里也包含了正样本
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)  # 所以后面直接算正样本就没问题的
    loss = -1 * mean_log_prob_pos.mean()
    return loss


def construct_data_pair(pl, logits, uncertainty, n_select_c=100, mode=1):
    # proportion of select c
    # positive hard pair   negative hard pair
    # pl.shape=[n]    logits.shape=[n, c]
    feas, labels = torch.tensor([]).cuda(), torch.tensor([]).cuda()
    unique_c = torch.unique(pl)
    # for c in unique_c:
    #     label_mask = pl == c
    #     uncertainty_mask = uncertainty[label_mask]
    #     uncertainty_mask_argsort = uncertainty_mask.argsort()[: n_select_c]
    #     print(uncertainty_mask[uncertainty_mask_argsort])
    #     fea_c = logits[uncertainty_mask_argsort]
    #     print(len(fea_c), label_mask.sum())

    # 只保留uncertainty低的features
    for c in unique_c:
        if c != 255:
            c_mask = pl == c
            label_mask = pl[c_mask]
            fea_mask = logits[c_mask]
            uncertainty_mask = uncertainty[c_mask]
            if mode == 1:
                uncertainty_mask_argsort = uncertainty_mask.argsort()[-n_select_c:]
                fea_mask = fea_mask[uncertainty_mask_argsort]
            feas = torch.cat((feas, fea_mask))
            labels = torch.cat((labels, label_mask[:len(fea_mask)]))

        elif mode == 2:
            # 同时考虑low uncertainty和
            pass

    feas = F.normalize(feas, p=2.0, dim=1)
    return feas, labels.unsqueeze(0)