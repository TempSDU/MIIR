import time
import random
import numpy as np
import scipy.sparse as sps
import torch
from transformers import *


#'''
#step 1
threshold = 5
features_map = {'category':set([]), 'brand':set([])}
users_list = {}
items_list = {}
interactions_list = {}
with open('raw/meta_Beauty.json', 'r', encoding='utf-8') as f:
    content = f.readlines()
for line in content:
    line = eval(line)
    item_id = line['asin']
    temp = []
    for i in range(len(line['categories'])):
        temp += line['categories'][i]
    temp = set(temp)
    category = list(temp)
    features_map['category'] = features_map['category']|temp
    if 'brand' in line:
        brand = line['brand']
        if brand != '':
            features_map['brand'].add(brand)
    else:
        brand = ''
    if 'title' in line:
        title = line['title']
    else:
        title = ''
    if 'description' in line:
        description = line['description']
    else:
        description = ''
    items_list[item_id] = {'item_id':item_id, 'category':category, 'brand':brand, 'title':title, 'description':description, 'users':set([])}
with open('raw/ratings_Beauty.csv', 'r', encoding='utf-8') as f:
    content = f.readlines()
for line in content:
    line = line.strip().split(',')
    user_id = line[0]
    item_id = line[1]
    timestamp = int(line[3])
    if user_id not in users_list:
        users_list[user_id] = {'user_id':user_id, 'items':set([])}
        interactions_list[user_id] = []
    if item_id not in items_list:
        items_list[item_id] = {'item_id':item_id, 'category':[], 'brand':'', 'title':'', 'description':'', 'users':set([])}  # note that '' means missing
    interactions_list[user_id].append((item_id, timestamp))
print('user num before filter:', len(users_list))
print('item num before filter:', len(items_list))
print('interaction num before filter:', len(content))
users_map = set([])
items_map = set([])
for user_id in list(users_list.keys()):
    if len(interactions_list[user_id]) >= threshold and len(interactions_list[user_id]) <= 1000:
        users_map.add(user_id)
        for interaction in interactions_list[user_id]:
            users_list[user_id]['items'].add(interaction[0])
            items_map.add(interaction[0])
            items_list[interaction[0]]['users'].add(user_id)
    else:
        interactions_list.pop(user_id)
        users_list.pop(user_id)
users_map = list(users_map)
users_map.sort()
items_map = list(items_map)
items_map.sort()
for item_id in list(items_list.keys()):
    if item_id not in items_map:
        items_list.pop(item_id)
num = 0
for user_id in users_map:
    interactions_list[user_id].sort(key=lambda x:x[1])
    num += len(interactions_list[user_id])
print('user num after filter:', len(users_map))
print('item num after filter:', len(items_map))
print('interaction num after filter:', num)
f1 = open('origin/interactions_list.dat', 'w')
f2 = open('origin/users_list.dat', 'w')
f3 = open('map/users_map.dat', 'w')
users_map_out = {'padding':0, 'missing':1}
reindex = 2
for user_id in users_map:
    for interaction in interactions_list[user_id]:
        f1.write(user_id+'|'+interaction[0]+'|'+str(interaction[1])+'\n')
    users_list[user_id]['items'] = list(users_list[user_id]['items'])
    f2.write(str(users_list[user_id])+'\n')
    users_map_out[user_id] = reindex
    reindex += 1
    f1.flush()
    f2.flush()
f3.write(str(users_map_out)+'\n')
f3.flush()
f1.close()
f2.close()
f3.close()
f1 = open('origin/items_list.dat', 'w')
f2 = open('map/items_map.dat', 'w')
items_map_out = {'padding':0, 'missing':1}
reindex = 2
for item_id in items_map:
    items_list[item_id]['users'] = list(items_list[item_id]['users'])
    f1.write(str(items_list[item_id])+'\n')
    items_map_out[item_id] = reindex
    reindex += 1
    f1.flush()
f2.write(str(items_map_out)+'\n')
f2.flush()
f1.close()
f2.close()
features_map_out = {}
f = open('map/features_map.dat', 'w')
for feature_field in features_map:
    features_map_out[feature_field] = {'missing':0}  # we will use one-hot vectors to represent features (the position of 1 is feature_id) excluding padding (we will use zero vector to represent padding)
    reindex = 1
    for feature in features_map[feature_field]:
        features_map_out[feature_field][feature] = reindex
        reindex += 1
f.write(str(features_map_out)+'\n')
f.flush()
f.close()
#'''


#'''
#step 2
users_list = []
with open('origin/users_list.dat', 'r') as f:
    content = f.readlines()
for line in content:
    line = eval(line.strip())
    users_list.append(line)
items_list = []
with open('origin/items_list.dat', 'r') as f:
    content = f.readlines()
for line in content:
    line = eval(line.strip())
    items_list.append(line)
with open('map/users_map.dat', 'r') as f:
    content = f.readlines()
users_map = eval(content[0].strip())
with open('map/items_map.dat', 'r') as f:
    content = f.readlines()
items_map = eval(content[0].strip())
with open('map/features_map.dat', 'r') as f:
    content = f.readlines()
features_map = eval(content[0].strip())
with open('origin/interactions_list.dat', 'r') as f:
    content = f.readlines()
interactions = np.zeros(len(content), dtype=np.int32)
with open('process/rows_file.dat', 'w') as f:
    start = 0
    end = 0
    i = 0
    for line in content:
        line = line.strip().split('|')
        user_id = line[0]
        item_id = line[1]
        if end == 0:
            pre_user_id = user_id
        if pre_user_id != user_id:
            f.write(str(users_map[pre_user_id])+'|'+str(start)+':'+str(end-1)+'\n')
            start = end
            pre_user_id = user_id
        end += 1
        interactions[i] = items_map[item_id]
        i += 1
    f.write(str(users_map[pre_user_id])+'|'+str(start)+':'+str(end-1)+'\n')
np.save('process/interactions_file.npy', interactions)
vals_c = [1]  # for missing, we set its category to missing vector (for padding, we set its category to zero vector)
rows_c = [1]
cols_c = [0]
vals_b = [1]  # for missing, we set its brand to missing vector (for padding, we set its brand to zero vector)
rows_b = [1]
cols_b = [0]
missings = np.zeros((len(items_list)+2, 5), dtype=np.int32)  # 5 -> item_id, category, brand, title, description, 0 means missing
missings[:, 0] = 1  # all items must have item_id
missings[[0,1], :] = 1  # for padding and missing
for item in items_list:
    item_id = items_map[item['item_id']]
    if len(item['category']) == 0:
        vals_c.append(1)
        rows_c.append(item_id)
        cols_c.append(0)
        missings[item_id, 1] = 0
    else:
        for cate in item['category']:
            vals_c.append(1)
            rows_c.append(item_id)
            cols_c.append(features_map['category'][cate])
        missings[item_id, 1] = 1
    if item['brand'] == '':
        vals_b.append(1)
        rows_b.append(item_id)
        cols_b.append(0)
        missings[item_id, 2] = 0
    else:
        vals_b.append(1)
        rows_b.append(item_id)
        cols_b.append(features_map['brand'][item['brand']])
        missings[item_id, 2] = 1
    if item['title'] == '':
        missings[item_id, 3] = 0
    else:
        missings[item_id, 3] = 1
    if item['description'] == '':
        missings[item_id, 4] = 0
    else:
        missings[item_id, 4] = 1
categories = sps.csr_matrix((vals_c, (rows_c, cols_c)), shape=(len(items_list)+2, len(features_map['category'])), dtype=np.int32)  # +2 is for padding and missing
brands = sps.csr_matrix((vals_b, (rows_b, cols_b)), shape=(len(items_list)+2, len(features_map['brand'])), dtype=np.int32)  # +2 is for padding and missing
sps.save_npz('process/categories_file.npz', categories)
sps.save_npz('process/brands_file.npz', brands)
np.save('process/missings_file.npy', missings)
#'''


#'''
#step 3
class Bert(torch.nn.Module):
    def __init__(self, output_layer, aggregator, pretrained_weights='bert-base-uncased'):
        super(Bert, self).__init__()
        assert 0<=output_layer<=13, '0 <= output_layer <= 13'
        assert aggregator in ['sum', 'mean', 'cls'], "aggregator in ['sum', 'mean', 'cls']"
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.model = BertModel.from_pretrained(pretrained_weights, output_hidden_states=True, output_attentions=True)
        self.output_layer = output_layer
        self.aggregator = aggregator

    def forward(self, input_ids, attention_mask):
        returns = self.model(input_ids, attention_mask)  # returns.last_hidden_state=[batch_size,seq_len,768](=returns.hidden_states[-1]), returns.pooler_output=[batch_size,768], returns.hidden_states=[13,batch_size,seq_len,768](returns.hidden_states[0]->embedding), returns.attentions=[12,batch_size,seq_len,seq_len]
        if self.output_layer < 13:
            if self.aggregator == 'sum':
                output = torch.sum(attention_mask.unsqueeze(-1)*returns.hidden_states[self.output_layer], 1)
            if self.aggregator == 'mean':
                output = torch.sum(attention_mask.unsqueeze(-1)*returns.hidden_states[self.output_layer], 1)/torch.sum(attention_mask, -1, keepdim=True)
            if self.aggregator == 'cls':
                output = returns.hidden_states[self.output_layer][:,0,:]
        if self.output_layer == 13:
            output = returns.pooler_output
        return output


with open('map/items_map.dat', 'r') as f:
    content = f.readlines()
items_map = eval(content[0].strip())
with open('origin/items_list.dat', 'r') as f:
    content = f.readlines()
titles = ['' for _ in range(len(content)+2)]  # +2 is for padding and missing, note that '' represents missing (the output of '' of Bert may have a little different)
descriptions = ['' for _ in range(len(content)+2)]  # +2 is for padding and missing, note that '' represents missing (the output of '' of Bert may have a little different)
for line in content:
    line = eval(line.strip())
    item_id = items_map[line['item_id']]
    titles[item_id] = line['title']
    descriptions[item_id] = line['description']
bert = Bert(12, 'cls')
bert.cuda()
bert.eval()
batch_size = 128
title_embeddings = []
with torch.no_grad():
    for i in range(int(np.ceil(len(titles)/batch_size))):
        start = i*batch_size
        end = (i+1)*batch_size
        batch = titles[start:end]
        tokens = bert.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        output = bert(input_ids, attention_mask)
        title_embeddings.append(output.detach().cpu().numpy())
title_embeddings = np.concatenate(title_embeddings)
title_embeddings[0] = 0  # for padding, we set its title embedding to zero vector (for missing, we set its title embedding to missing vector)
np.save('process/titles_file.npy', title_embeddings)
batch_size = 64
description_embeddings = []
with torch.no_grad():
    for i in range(int(np.ceil(len(descriptions)/batch_size))):
        start = i*batch_size
        end = (i+1)*batch_size
        batch = descriptions[start:end]
        tokens = bert.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokens['input_ids'].cuda()
        attention_mask = tokens['attention_mask'].cuda()
        output = bert(input_ids, attention_mask)
        description_embeddings.append(output.detach().cpu().numpy())
description_embeddings = np.concatenate(description_embeddings)
description_embeddings[0] = 0  # for padding, we set its description embedding to zero vector (for missing, we set its description embedding to missing vector)
np.save('process/descriptions_file.npy', description_embeddings)
#'''


#'''
#step 4
missings = np.load('process/missings_file.npy')
masks = np.random.rand(missings.shape[0], missings.shape[1])  # we need mask more side information, 1 means mask
masks = (masks < 0.5).astype('int32')
masks[:, 0] = 0  # for item id
masks[[0,1], :] = 0  # for padding and missing
masks = missings*masks
np.save('process/masks_file.npy', masks)
max_interactions = 20
with open('process/rows_file.dat', 'r') as f:
    content = f.readlines()
f1 = open('process/train_rows_file.dat', 'w')
f2 = open('process/valid_rows_file.dat', 'w')
f3 = open('process/test_rows_file.dat', 'w')
for line in content:
    line = line.strip().split('|')
    user_id = line[0]
    start, end = line[1].split(':')
    if int(end)-1-max_interactions+1 >= int(start):
        f2.write(user_id+'|'+str(int(end)-1-max_interactions+1)+':'+str(int(end)-1)+'\n')
    else:
        f2.write(user_id+'|'+str(int(start))+':'+str(int(end)-1)+'\n')
    if int(end)-max_interactions+1 >= int(start):
        f3.write(user_id+'|'+str(int(end)-max_interactions+1)+':'+str(int(end))+'\n')
    else:
        f3.write(user_id+'|'+str(int(start))+':'+str(int(end))+'\n')
    indices = list(range(int(start), int(end)-1, 1))
    for i in range(0, len(indices), max_interactions):
        start_i = indices[i:i+max_interactions][0]
        end_i = indices[i:i+max_interactions][-1]
        f1.write(user_id+'|'+str(start_i)+':'+str(end_i)+'\n')
    f1.flush()
    f2.flush()
    f3.flush()
f1.close()
f2.close()
f3.close()
interactions = np.load('process/interactions_file.npy')
with open('map/items_map.dat', 'r') as f:
    content = f.readlines()
allitems = eval(content[0].strip())
allitems = set(range(len(allitems)))-{0,1}
with open('process/valid_rows_file.dat', 'r') as f:
    content = f.readlines()
valid_negatives = np.zeros((len(content), 99), dtype=np.int32)
i = 0
for line in content:
    line = line.strip().split('|')
    user_id = line[0]
    start, end = line[1].split(':')
    negatives = random.sample(allitems-{interactions[int(end)]}, 99)
    valid_negatives[i] = np.array(negatives)
    i += 1
np.save('process/valid_negatives_file.npy', valid_negatives)
with open('process/test_rows_file.dat', 'r') as f:
    content = f.readlines()
test_negatives = np.zeros((len(content), 99), dtype=np.int32)
i = 0
for line in content:
    line = line.strip().split('|')
    user_id = line[0]
    start, end = line[1].split(':')
    negatives = random.sample(allitems-{interactions[int(end)]}, 99)
    test_negatives[i] = np.array(negatives)
    i += 1
np.save('process/test_negatives_file.npy', test_negatives)
#'''
