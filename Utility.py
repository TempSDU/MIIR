import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


def xavier_init(net):
    for name, par in net.named_parameters():
        if 'weight' in name and len(par.shape) >= 2:
            nn.init.xavier_normal_(par)
        elif 'bias' in name:
            nn.init.constant_(par, 0.0)


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def evaluatefunction(output, positives, negatives):
    result = [{} for _ in range(len(output))]
    for i in range(len(output)):
        pos_score = output[i][positives[i]]  # 1
        neg_scores = output[i][negatives[i]]  # 99
        success = ((neg_scores-pos_score) < 0).sum()
        if 99-success < 5:
            result[i]['recall@5'] = 1
        else:
            result[i]['recall@5'] = 0
        if 99-success < 10:
            result[i]['recall@10'] = 1
        else:
            result[i]['recall@10'] = 0
        result[i]['auc'] = success/99
        result[i]['mrr'] = 1/(99-success+1)
    return result
