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


'''
def evaluatefunction2(output, positives, negatives, target, mask, feature_field):
    result = [{} for _ in range(len(output))]
    if feature_field == 'id':
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
    if feature_field == 'category':
        for i in range(len(output)):
            pre_category = output[i]  # [seq_len, category_num]
            pre_category = (pre_category > 0).astype('int32')
            tar_category = target[i]  # [seq_len, category_num]
            over_category = pre_category*tar_category  # [seq_len, category_num]
            pre_category = pre_category.sum(-1)  # [seq_len]
            tar_category = tar_category.sum(-1)  # [seq_len]
            over_category = over_category.sum(-1)  # [seq_len]
            result[i]['precision'] = (over_category/(pre_category+(pre_category==0).astype('int32'))*mask[i]).sum()/(mask[i].sum()+(mask[i].sum()==0).astype('int32'))
            result[i]['recall'] = (over_category/(tar_category+(tar_category==0).astype('int32'))*mask[i]).sum()/(mask[i].sum()+(mask[i].sum()==0).astype('int32'))
            result[i]['f1'] = 2*result[i]['precision']*result[i]['recall']/(result[i]['precision']+result[i]['recall']+((result[i]['precision']+result[i]['recall'])==0).astype('int32'))
    if feature_field == 'brand':
        for i in range(len(output)):
            pre_brand = output[i]  # [seq_len, brand_num]
            tar_brand = target[i]  # [seq_len, brand_num]
            pre_brand = pre_brand.argmax(-1)  # [seq_len]
            tar_brand = tar_brand.argmax(-1)  # [seq_len]
            over_brand = (pre_brand == tar_brand).astype('int32')
            result[i]['accuracy'] = (over_brand*mask[i]).sum()/(mask[i].sum()+(mask[i].sum()==0).astype('int32'))
    if feature_field == 'title':
        for i in range(len(output)):
            pre_title = output[i]  # [seq_len, 768]
            tar_title = target[i]  # [seq_len, 768]
            result[i]['mse'] = (((pre_title-tar_title)*(pre_title-tar_title)).sum(-1)/768*mask[i]).sum()/(mask[i].sum()+(mask[i].sum()==0).astype('int32'))
    if feature_field == 'description':
        for i in range(len(output)):
            pre_description = output[i]  # [seq_len, 768]
            tar_description = target[i]  # [seq_len, 768]
            result[i]['mse'] = (((pre_description-tar_description)*(pre_description-tar_description)).sum(-1)/768*mask[i]).sum()/(mask[i].sum()+(mask[i].sum()==0).astype('int32'))
    return result
'''
