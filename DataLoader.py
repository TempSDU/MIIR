import os
import shutil
import random
import numpy as np
import scipy.sparse as sps
import torch
from torch.utils import data
from torch.utils.data import DataLoader


class TrainDataset_MII(data.Dataset):
    def __init__(self, interactions_filepath, categories_filepath, brands_filepath, titles_filepath, descriptions_filepath, missings_filepath, rows_filepath, max_interactions, mask_prob, masks_filepath=None):
        super().__init__()
        self.interactions = np.load(interactions_filepath)  # [interaction_num]
        self.categories = sps.load_npz(categories_filepath)  # [item_num, category_num], note that include padding and missing
        self.brands = sps.load_npz(brands_filepath)  # [item_num, brand_num], note that include padding and missing
        self.titles = np.load(titles_filepath)  # [item_num, 768], note that include padding and missing
        self.descriptions = np.load(descriptions_filepath)  # [item_num, 768], note that include padding and missing
        self.missings = np.load(missings_filepath)  # [item_num, 5], note that include padding and missing, 0 means missing
        self.rows = []  # [dataset_size]
        with open(rows_filepath, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip().split('|')
            start, end = line[1].split(':')
            self.rows.append((int(line[0]), int(start), int(end)))
        self.max_interactions = max_interactions
        self.mask_prob = mask_prob
        if masks_filepath:
            masks = np.load(masks_filepath)  # [item_num, 5], note that include padding and missing, 1 means mask
            self.missings = self.missings-masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        user, start, end = self.rows[index]
        indices = list(range(start, end+1, 1))
        output_session_ids = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions]
        output_session_ids[:len(indices)] = self.interactions[indices]
        session_missings = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5]
        session_missings[:] = self.missings[output_session_ids]
        output_session_categories = np.zeros((self.max_interactions, self.categories.shape[1]), dtype=np.int32)  # [max_interactions, category_num]
        output_session_categories[:] = self.categories[output_session_ids].toarray()
        output_session_categories[session_missings[:,1] == 0, :] = 0
        output_session_categories[session_missings[:,1] == 0, 0] = 1
        output_session_brands = np.zeros((self.max_interactions, self.brands.shape[1]), dtype=np.int32)  # [max_interactions, brand_num]
        output_session_brands[:] = self.brands[output_session_ids].toarray()
        output_session_brands[session_missings[:,2] == 0, :] = 0
        output_session_brands[session_missings[:,2] == 0, 0] = 1
        output_session_titles = np.zeros((self.max_interactions, self.titles.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        output_session_titles[:] = self.titles[output_session_ids]
        output_session_titles[session_missings[:,3] == 0, :] = self.titles[1]
        output_session_descriptions = np.zeros((self.max_interactions, self.descriptions.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        output_session_descriptions[:] = self.descriptions[output_session_ids]
        output_session_descriptions[session_missings[:,4] == 0, :] = self.descriptions[1]
        padding_mask = np.array([0]*len(indices)+[1]*(self.max_interactions-len(indices)), dtype=np.int32)  # [max_interactions], 1 represents padding
        loss_mask = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5], 1 represents loss
        loss_mask[np.random.rand(self.max_interactions, 5) < self.mask_prob] = 1
        loss_mask[:,0] = loss_mask[:,0]*(1-padding_mask)
        input_session_ids = output_session_ids.copy()
        input_session_ids[loss_mask[:,0] == 1] = 1
        loss_mask[:,1] = loss_mask[:,1]*session_missings[:,1]*(1-padding_mask)
        input_session_categories = output_session_categories.copy()
        input_session_categories[loss_mask[:,1] == 1, :] = 0
        input_session_categories[loss_mask[:,1] == 1, 0] = 1
        loss_mask[:,2] = loss_mask[:,2]*session_missings[:,2]*(1-padding_mask)
        input_session_brands = output_session_brands.copy()
        input_session_brands[loss_mask[:,2] == 1, :] = 0
        input_session_brands[loss_mask[:,2] == 1, 0] = 1
        loss_mask[:,3] = loss_mask[:,3]*session_missings[:,3]*(1-padding_mask)
        input_session_titles = output_session_titles.copy()
        input_session_titles[loss_mask[:,3] == 1, :] = self.titles[1]
        loss_mask[:,4] = loss_mask[:,4]*session_missings[:,4]*(1-padding_mask)
        input_session_descriptions = output_session_descriptions.copy()
        input_session_descriptions[loss_mask[:,4] == 1, :] = self.descriptions[1]
        return torch.tensor(index).int(), [torch.tensor(input_session_ids).long(), torch.tensor(input_session_categories).float(), torch.tensor(input_session_brands).float(), torch.tensor(input_session_titles).float(), torch.tensor(input_session_descriptions).float()], [torch.tensor(output_session_ids).long(), torch.tensor(output_session_categories).float(), torch.tensor(output_session_brands).float(), torch.tensor(output_session_titles).float(), torch.tensor(output_session_descriptions).float()], [torch.tensor(padding_mask).bool(), torch.tensor(loss_mask).float()]


class TrainDataset_MLM(data.Dataset):
    def __init__(self, interactions_filepath, categories_filepath, brands_filepath, titles_filepath, descriptions_filepath, missings_filepath, rows_filepath, max_interactions, mask_prob, masks_filepath=None):
        super().__init__()
        self.interactions = np.load(interactions_filepath)  # [interaction_num]
        self.categories = sps.load_npz(categories_filepath)  # [item_num, category_num], note that include padding and missing
        self.brands = sps.load_npz(brands_filepath)  # [item_num, brand_num], note that include padding and missing
        self.titles = np.load(titles_filepath)  # [item_num, 768], note that include padding and missing
        self.descriptions = np.load(descriptions_filepath)  # [item_num, 768], note that include padding and missing
        self.missings = np.load(missings_filepath)  # [item_num, 5], note that include padding and missing, 0 means missing
        self.rows = []  # [dataset_size]
        with open(rows_filepath, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip().split('|')
            start, end = line[1].split(':')
            self.rows.append((int(line[0]), int(start), int(end)))
        self.max_interactions = max_interactions
        self.mask_prob = mask_prob
        if masks_filepath:
            masks = np.load(masks_filepath)  # [item_num, 5], note that include padding and missing, 1 means mask
            self.missings = self.missings-masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        user, start, end = self.rows[index]
        indices = list(range(start, end+1, 1))
        output_session_ids = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions]
        output_session_ids[:len(indices)] = self.interactions[indices]
        padding_mask = np.array([0]*len(indices)+[1]*(self.max_interactions-len(indices)), dtype=np.int32)  # [max_interactions], 1 represents padding
        loss_mask = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions], 1 represents loss
        if np.random.rand(1) > 0.0:  # randomly mask
            loss_mask[np.random.rand(self.max_interactions) < self.mask_prob] = 1
            loss_mask *= 1-padding_mask
        else:  # only mask the last one
            loss_mask[len(indices)-1] = 1
        input_session_ids = output_session_ids.copy()
        input_session_ids[loss_mask == 1] = 1
        session_missings = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5]
        session_missings[:] = self.missings[input_session_ids]
        session_categories = np.zeros((self.max_interactions, self.categories.shape[1]), dtype=np.int32)  # [max_interactions, category_num]
        session_categories[:] = self.categories[input_session_ids].toarray()
        session_categories[session_missings[:,1] == 0, :] = 0
        session_categories[session_missings[:,1] == 0, 0] = 1
        session_brands = np.zeros((self.max_interactions, self.brands.shape[1]), dtype=np.int32)  # [max_interactions, brand_num]
        session_brands[:] = self.brands[input_session_ids].toarray()
        session_brands[session_missings[:,2] == 0, :] = 0
        session_brands[session_missings[:,2] == 0, 0] = 1
        session_titles = np.zeros((self.max_interactions, self.titles.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_titles[:] = self.titles[input_session_ids]
        session_titles[session_missings[:,3] == 0, :] = self.titles[1]
        session_descriptions = np.zeros((self.max_interactions, self.descriptions.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_descriptions[:] = self.descriptions[input_session_ids]
        session_descriptions[session_missings[:,4] == 0, :] = self.descriptions[1]
        return torch.tensor(index).int(), [torch.tensor(input_session_ids).long(), torch.tensor(output_session_ids).long()], [torch.tensor(session_categories).float(), torch.tensor(session_brands).float(), torch.tensor(session_titles).float(), torch.tensor(session_descriptions).float()], [torch.tensor(padding_mask).bool(), torch.tensor(loss_mask).float()]


class TestDataset_MLM(data.Dataset):
    def __init__(self, interactions_filepath, categories_filepath, brands_filepath, titles_filepath, descriptions_filepath, missings_filepath, rows_filepath, negatives_filepath, max_interactions, masks_filepath=None):
        super().__init__()
        self.interactions = np.load(interactions_filepath)  # [interaction_num]
        self.categories = sps.load_npz(categories_filepath)  # [item_num, category_num], note that include padding and missing
        self.brands = sps.load_npz(brands_filepath)  # [item_num, brand_num], note that include padding and missing
        self.titles = np.load(titles_filepath)  # [item_num, 768], note that include padding and missing
        self.descriptions = np.load(descriptions_filepath)  # [item_num, 768], note that include padding and missing
        self.missings = np.load(missings_filepath)  # [item_num, 5], note that include padding and missing, 0 means missing
        self.rows = []  # [dataset_size]
        with open(rows_filepath, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip().split('|')
            start, end = line[1].split(':')
            self.rows.append((int(line[0]), int(start), int(end)))
        self.negatives = np.load(negatives_filepath)  # [dataset_size, 99]
        self.max_interactions = max_interactions
        if masks_filepath:
            masks = np.load(masks_filepath)  # [item_num, 5], note that include padding and missing, 1 means mask
            self.missings = self.missings-masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        user, start, end = self.rows[index]
        indices = list(range(start, end+1, 1))
        output_session_ids = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions]
        output_session_ids[:len(indices)] = self.interactions[indices]
        padding_mask = np.array([0]*len(indices)+[1]*(self.max_interactions-len(indices)), dtype=np.int32)  # [max_interactions], 1 represents padding
        loss_mask = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions], 1 represents loss
        loss_mask[len(indices)-1] = 1
        input_session_ids = output_session_ids.copy()
        input_session_ids[loss_mask == 1] = 1
        session_missings = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5]
        session_missings[:] = self.missings[input_session_ids]
        session_categories = np.zeros((self.max_interactions, self.categories.shape[1]), dtype=np.int32)  # [max_interactions, category_num]
        session_categories[:] = self.categories[input_session_ids].toarray()
        session_categories[session_missings[:,1] == 0, :] = 0
        session_categories[session_missings[:,1] == 0, 0] = 1
        session_brands = np.zeros((self.max_interactions, self.brands.shape[1]), dtype=np.int32)  # [max_interactions, brand_num]
        session_brands[:] = self.brands[input_session_ids].toarray()
        session_brands[session_missings[:,2] == 0, :] = 0
        session_brands[session_missings[:,2] == 0, 0] = 1
        session_titles = np.zeros((self.max_interactions, self.titles.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_titles[:] = self.titles[input_session_ids]
        session_titles[session_missings[:,3] == 0, :] = self.titles[1]
        session_descriptions = np.zeros((self.max_interactions, self.descriptions.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_descriptions[:] = self.descriptions[input_session_ids]
        session_descriptions[session_missings[:,4] == 0, :] = self.descriptions[1]
        session_negatives = self.negatives[index]  # [99]
        return torch.tensor(index).int(), [torch.tensor(input_session_ids).long(), torch.tensor(output_session_ids).long()], [torch.tensor(session_categories).float(), torch.tensor(session_brands).float(), torch.tensor(session_titles).float(), torch.tensor(session_descriptions).float()], [torch.tensor(padding_mask).bool(), torch.tensor(loss_mask).float()], torch.tensor(session_negatives).int()


'''
class TrainDataset_MII(data.Dataset):  # treat original missing feature fields as paddings in self-attention
    def __init__(self, interactions_filepath, categories_filepath, brands_filepath, titles_filepath, descriptions_filepath, missings_filepath, rows_filepath, max_interactions, mask_prob, masks_filepath=None):
        super().__init__()
        self.interactions = np.load(interactions_filepath)  # [interaction_num]
        self.categories = sps.load_npz(categories_filepath)  # [item_num, category_num], note that include padding and missing
        self.brands = sps.load_npz(brands_filepath)  # [item_num, brand_num], note that include padding and missing
        self.titles = np.load(titles_filepath)  # [item_num, 768], note that include padding and missing
        self.descriptions = np.load(descriptions_filepath)  # [item_num, 768], note that include padding and missing
        self.missings = np.load(missings_filepath)  # [item_num, 5], note that include padding and missing, 0 means missing
        self.rows = []  # [dataset_size]
        with open(rows_filepath, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip().split('|')
            start, end = line[1].split(':')
            self.rows.append((int(line[0]), int(start), int(end)))
        self.max_interactions = max_interactions
        self.mask_prob = mask_prob
        if masks_filepath:
            masks = np.load(masks_filepath)  # [item_num, 5], note that include padding and missing, 1 means mask
            self.missings = self.missings-masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        user, start, end = self.rows[index]
        indices = list(range(start, end+1, 1))
        output_session_ids = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions]
        output_session_ids[:len(indices)] = self.interactions[indices]
        session_missings = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5]
        session_missings[:] = self.missings[output_session_ids]
        output_session_categories = np.zeros((self.max_interactions, self.categories.shape[1]), dtype=np.int32)  # [max_interactions, category_num]
        output_session_categories[:] = self.categories[output_session_ids].toarray()
        output_session_categories[session_missings[:,1] == 0, :] = 0
        output_session_categories[session_missings[:,1] == 0, 0] = 1
        output_session_brands = np.zeros((self.max_interactions, self.brands.shape[1]), dtype=np.int32)  # [max_interactions, brand_num]
        output_session_brands[:] = self.brands[output_session_ids].toarray()
        output_session_brands[session_missings[:,2] == 0, :] = 0
        output_session_brands[session_missings[:,2] == 0, 0] = 1
        output_session_titles = np.zeros((self.max_interactions, self.titles.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        output_session_titles[:] = self.titles[output_session_ids]
        output_session_titles[session_missings[:,3] == 0, :] = self.titles[1]
        output_session_descriptions = np.zeros((self.max_interactions, self.descriptions.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        output_session_descriptions[:] = self.descriptions[output_session_ids]
        output_session_descriptions[session_missings[:,4] == 0, :] = self.descriptions[1]
        padding_mask = np.array([0]*len(indices)+[1]*(self.max_interactions-len(indices)), dtype=np.int32)  # [max_interactions], 1 represents padding
        loss_mask = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5], 1 represents loss
        loss_mask[np.random.rand(self.max_interactions, 5) < self.mask_prob] = 1
        loss_mask[:,0] = loss_mask[:,0]*(1-padding_mask)
        input_session_ids = output_session_ids.copy()
        input_session_ids[loss_mask[:,0] == 1] = 1
        loss_mask[:,1] = loss_mask[:,1]*session_missings[:,1]*(1-padding_mask)
        input_session_categories = output_session_categories.copy()
        input_session_categories[loss_mask[:,1] == 1, :] = 0
        input_session_categories[loss_mask[:,1] == 1, 0] = 1
        loss_mask[:,2] = loss_mask[:,2]*session_missings[:,2]*(1-padding_mask)
        input_session_brands = output_session_brands.copy()
        input_session_brands[loss_mask[:,2] == 1, :] = 0
        input_session_brands[loss_mask[:,2] == 1, 0] = 1
        loss_mask[:,3] = loss_mask[:,3]*session_missings[:,3]*(1-padding_mask)
        input_session_titles = output_session_titles.copy()
        input_session_titles[loss_mask[:,3] == 1, :] = self.titles[1]
        loss_mask[:,4] = loss_mask[:,4]*session_missings[:,4]*(1-padding_mask)
        input_session_descriptions = output_session_descriptions.copy()
        input_session_descriptions[loss_mask[:,4] == 1, :] = self.descriptions[1]
        padding_mask = 1-session_missings  # [max_interactions, 5], 1 means mask
        padding_mask[len(indices):, :] = 1
        return torch.tensor(index).int(), [torch.tensor(input_session_ids).long(), torch.tensor(input_session_categories).float(), torch.tensor(input_session_brands).float(), torch.tensor(input_session_titles).float(), torch.tensor(input_session_descriptions).float()], [torch.tensor(output_session_ids).long(), torch.tensor(output_session_categories).float(), torch.tensor(output_session_brands).float(), torch.tensor(output_session_titles).float(), torch.tensor(output_session_descriptions).float()], [torch.tensor(padding_mask).bool(), torch.tensor(loss_mask).float()]
'''
'''
class TrainDataset_MLM(data.Dataset):  # treat original missing feature fields as paddings in self-attention
    def __init__(self, interactions_filepath, categories_filepath, brands_filepath, titles_filepath, descriptions_filepath, missings_filepath, rows_filepath, max_interactions, mask_prob, masks_filepath=None):
        super().__init__()
        self.interactions = np.load(interactions_filepath)  # [interaction_num]
        self.categories = sps.load_npz(categories_filepath)  # [item_num, category_num], note that include padding and missing
        self.brands = sps.load_npz(brands_filepath)  # [item_num, brand_num], note that include padding and missing
        self.titles = np.load(titles_filepath)  # [item_num, 768], note that include padding and missing
        self.descriptions = np.load(descriptions_filepath)  # [item_num, 768], note that include padding and missing
        self.missings = np.load(missings_filepath)  # [item_num, 5], note that include padding and missing, 0 means missing
        self.rows = []  # [dataset_size]
        with open(rows_filepath, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip().split('|')
            start, end = line[1].split(':')
            self.rows.append((int(line[0]), int(start), int(end)))
        self.max_interactions = max_interactions
        self.mask_prob = mask_prob
        if masks_filepath:
            masks = np.load(masks_filepath)  # [item_num, 5], note that include padding and missing, 1 means mask
            self.missings = self.missings-masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        user, start, end = self.rows[index]
        indices = list(range(start, end+1, 1))
        output_session_ids = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions]
        output_session_ids[:len(indices)] = self.interactions[indices]
        padding_mask = np.array([0]*len(indices)+[1]*(self.max_interactions-len(indices)), dtype=np.int32)  # [max_interactions], 1 represents padding
        loss_mask = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions], 1 represents loss
        if np.random.rand(1) > 0.0:  # randomly mask
            loss_mask[np.random.rand(self.max_interactions) < self.mask_prob] = 1
            loss_mask *= 1-padding_mask
        else:  # only mask the last one
            loss_mask[len(indices)-1] = 1
        input_session_ids = output_session_ids.copy()
        input_session_ids[loss_mask == 1] = 1
        session_missings = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5]
        session_missings[:] = self.missings[input_session_ids]
        session_categories = np.zeros((self.max_interactions, self.categories.shape[1]), dtype=np.int32)  # [max_interactions, category_num]
        session_categories[:] = self.categories[input_session_ids].toarray()
        session_categories[session_missings[:,1] == 0, :] = 0
        session_categories[session_missings[:,1] == 0, 0] = 1
        session_brands = np.zeros((self.max_interactions, self.brands.shape[1]), dtype=np.int32)  # [max_interactions, brand_num]
        session_brands[:] = self.brands[input_session_ids].toarray()
        session_brands[session_missings[:,2] == 0, :] = 0
        session_brands[session_missings[:,2] == 0, 0] = 1
        session_titles = np.zeros((self.max_interactions, self.titles.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_titles[:] = self.titles[input_session_ids]
        session_titles[session_missings[:,3] == 0, :] = self.titles[1]
        session_descriptions = np.zeros((self.max_interactions, self.descriptions.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_descriptions[:] = self.descriptions[input_session_ids]
        session_descriptions[session_missings[:,4] == 0, :] = self.descriptions[1]
        padding_mask = 1-session_missings  # [max_interactions, 5], 1 means mask
        padding_mask[len(indices):, :] = 1
        padding_mask[loss_mask == 1, 1:] = 1
        return torch.tensor(index).int(), [torch.tensor(input_session_ids).long(), torch.tensor(output_session_ids).long()], [torch.tensor(session_categories).float(), torch.tensor(session_brands).float(), torch.tensor(session_titles).float(), torch.tensor(session_descriptions).float()], [torch.tensor(padding_mask).bool(), torch.tensor(loss_mask).float()]
'''
'''
class TestDataset_MLM(data.Dataset):  # treat original missing feature fields as paddings in self-attention
    def __init__(self, interactions_filepath, categories_filepath, brands_filepath, titles_filepath, descriptions_filepath, missings_filepath, rows_filepath, negatives_filepath, max_interactions, masks_filepath=None):
        super().__init__()
        self.interactions = np.load(interactions_filepath)  # [interaction_num]
        self.categories = sps.load_npz(categories_filepath)  # [item_num, category_num], note that include padding and missing
        self.brands = sps.load_npz(brands_filepath)  # [item_num, brand_num], note that include padding and missing
        self.titles = np.load(titles_filepath)  # [item_num, 768], note that include padding and missing
        self.descriptions = np.load(descriptions_filepath)  # [item_num, 768], note that include padding and missing
        self.missings = np.load(missings_filepath)  # [item_num, 5], note that include padding and missing, 0 means missing
        self.rows = []  # [dataset_size]
        with open(rows_filepath, 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.strip().split('|')
            start, end = line[1].split(':')
            self.rows.append((int(line[0]), int(start), int(end)))
        self.negatives = np.load(negatives_filepath)  # [dataset_size, 99]
        self.max_interactions = max_interactions
        if masks_filepath:
            masks = np.load(masks_filepath)  # [item_num, 5], note that include padding and missing, 1 means mask
            self.missings = self.missings-masks

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        user, start, end = self.rows[index]
        indices = list(range(start, end+1, 1))
        output_session_ids = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions]
        output_session_ids[:len(indices)] = self.interactions[indices]
        loss_mask = np.zeros(self.max_interactions, dtype=np.int32)  # [max_interactions], 1 represents loss
        loss_mask[len(indices)-1] = 1
        input_session_ids = output_session_ids.copy()
        input_session_ids[loss_mask == 1] = 1
        session_missings = np.zeros((self.max_interactions, 5), dtype=np.int32)  # [max_interactions, 5]
        session_missings[:] = self.missings[input_session_ids]
        session_categories = np.zeros((self.max_interactions, self.categories.shape[1]), dtype=np.int32)  # [max_interactions, category_num]
        session_categories[:] = self.categories[input_session_ids].toarray()
        session_categories[session_missings[:,1] == 0, :] = 0
        session_categories[session_missings[:,1] == 0, 0] = 1
        session_brands = np.zeros((self.max_interactions, self.brands.shape[1]), dtype=np.int32)  # [max_interactions, brand_num]
        session_brands[:] = self.brands[input_session_ids].toarray()
        session_brands[session_missings[:,2] == 0, :] = 0
        session_brands[session_missings[:,2] == 0, 0] = 1
        session_titles = np.zeros((self.max_interactions, self.titles.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_titles[:] = self.titles[input_session_ids]
        session_titles[session_missings[:,3] == 0, :] = self.titles[1]
        session_descriptions = np.zeros((self.max_interactions, self.descriptions.shape[1]), dtype=np.float32)  # [max_interactions, 768]
        session_descriptions[:] = self.descriptions[input_session_ids]
        session_descriptions[session_missings[:,4] == 0, :] = self.descriptions[1]
        session_negatives = self.negatives[index]  # [99]
        padding_mask = 1-session_missings  # [max_interactions, 5], 1 means mask
        padding_mask[len(indices):, :] = 1
        padding_mask[len(indices)-1, 1:] = 1
        return torch.tensor(index).int(), [torch.tensor(input_session_ids).long(), torch.tensor(output_session_ids).long()], [torch.tensor(session_categories).float(), torch.tensor(session_brands).float(), torch.tensor(session_titles).float(), torch.tensor(session_descriptions).float()], [torch.tensor(padding_mask).bool(), torch.tensor(loss_mask).float()], torch.tensor(session_negatives).int()
'''
