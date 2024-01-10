import torch
import torch.functional as F


class HingeLoss(torch.nn.Module):
    """Hinge Loss for pairwise learning.
    reference: https://github.com/ustcml/RecStudio/blob/main/recstudio/model/loss_func.py

    """

    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, pos_score, neg_score):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score + self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            return torch.mean(loss * torch.log(rank + 1))
        else:
            return torch.mean(loss)


class BPRLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        loss = torch.mean(-(pos_score - neg_score).sigmoid().log(), dim=-1)
        return loss
        #loss = -torch.mean(F.logsigmoid(pos_score - torch.max(neg_score, dim=-1))) need v1.10

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Caulate Gram core matrix
    source: data with sample_size_1 * feature_size
    target: data with sample_size_2 * feature_size 
    kernel_mul: core bandwith
    kernel_num: core number
    fix_sigma: fixed sigma or not
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # calculate |x-y| in Gaussian core

    # calculate each core bandwidth of multi-core
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # gaussian core: exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # combine multi-core