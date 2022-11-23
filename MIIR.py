import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer


class MIIRS(nn.Module):
    def __init__(self, dataset, emb_size=64, layer_num=3):
        super(MIIRS, self).__init__()
        self.emb_size = emb_size
        self.layer_num = layer_num
        if dataset == 'tg':
            self.item_num = 164980  # note that include padding and missing
            self.feature_fields = [(958, 'sigmoid'),  # category, note that include missing
                                   (14136, 'softmax'),  # brand, note that include missing
                                   (768, None),  # title
                                   (768, None)]  # description
        if dataset == 'bt':
            self.item_num = 121293  # note that include padding and missing
            self.feature_fields = [(657, 'sigmoid'),  # category, note that include missing
                                   (13189, 'softmax'),  # brand, note that include missing
                                   (768, None),  # title
                                   (768, None)]  # description
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.position_embeddings = nn.Embedding(100, self.emb_size)
        self.feature_field_embeddings = nn.Embedding(len(self.feature_fields)+1, self.emb_size)  # +1 for item id
        self.input_mlps = nn.ModuleList()
        self.output_mlps = nn.ModuleList()
        for feature_field in self.feature_fields:
            if feature_field[1]:
                feature_embeddings = nn.Embedding(feature_field[0], self.emb_size)
                self.input_mlps.append(feature_embeddings)
                self.output_mlps.append(feature_embeddings)
            else:
                self.input_mlps.append(nn.Linear(feature_field[0], self.emb_size))
                self.output_mlps.append(nn.Linear(self.emb_size, feature_field[0]))
        self.net = nn.ModuleList(TransformerEncoderLayer(d_model=self.emb_size, nhead=4, dim_feedforward=self.emb_size*4, dropout=0.5, activation='gelu') for _ in range(self.layer_num))  # note that d_model%nhead=0
        self.cross_mask = None

    def generate_cross_mask(self, seq_len, field_num):
        if self.cross_mask == None:
            cross_mask = torch.ones((seq_len*field_num, seq_len*field_num), device=self.item_embeddings.weight.device)  # 1/True is mask, 0/False is unmask
            i = 0
            while i < seq_len*field_num:
                cross_mask[i, i//field_num*field_num:(i//field_num+1)*field_num] = 0  # heterogeneous field
                cross_mask[i, range(i%field_num, seq_len*field_num, field_num)] = 0  # homogeneous field
                i += 1
            self.cross_mask = cross_mask.bool()
        else:
            if self.cross_mask.shape[0] != seq_len*field_num:
                cross_mask = torch.ones((seq_len*field_num, seq_len*field_num), device=self.item_embeddings.weight.device)  # 1/True is mask, 0/False is unmask
                i = 0
                while i < seq_len*field_num:
                    cross_mask[i, i//field_num*field_num:(i//field_num+1)*field_num] = 0  # heterogeneous field
                    cross_mask[i, range(i%field_num, seq_len*field_num, field_num)] = 0  # homogeneous field
                    i += 1
                self.cross_mask = cross_mask.bool()
        return self.cross_mask

    def forward(self, input_session_ids, session_feature_fields, padding_mask):
        item_embs = self.item_embeddings(input_session_ids)  # [batch_size, seq_len, emb_size]
        item_embs = item_embs.unsqueeze(2)  # [batch_size, seq_len, 1, emb_size]
        inputs = [item_embs]
        f = 0
        for feature_field in session_feature_fields:
            if self.feature_fields[f][1]:
                feature_embs = torch.einsum('ijl,lk->ijk', feature_field, self.input_mlps[f].weight)  # [batch_size, seq_len, emb_size]
            else:
                feature_embs = self.input_mlps[f](feature_field)  # [batch_size, seq_len, emb_size]
            feature_embs = feature_embs.unsqueeze(2)  # [batch_size, seq_len, 1, emb_size]
            inputs.append(feature_embs)
            f += 1
        inputs = torch.cat(inputs, 2)  # [batch_size, seq_len, field_num, emb_size]
        pos_ids = torch.arange(0, item_embs.shape[1], device=inputs.device).long()  # [seq_len]
        pos_embs = self.position_embeddings(pos_ids)  # [seq_len, emb_size]
        pos_embs = pos_embs.unsqueeze(0)  # [1, seq_len, emb_size]
        pos_embs = pos_embs.unsqueeze(2)  # [1, seq_len, 1, emb_size]
        inputs += pos_embs
        field_ids = torch.arange(0, len(self.feature_fields)+1, device=inputs.device).long()  # [field_num]
        field_embs = self.feature_field_embeddings(field_ids)  # [field_num, emb_size]
        field_embs = field_embs.unsqueeze(0)  # [1, field_num, emb_size]
        field_embs = field_embs.unsqueeze(1)  # [1, 1, field_num, emb_size]
        inputs += field_embs
        shape = inputs.shape
        inputs = inputs.reshape(shape[0], shape[1]*shape[2], shape[3])  # [batch_size, seq_len*field_num, emb_size]
        temps = inputs.permute(1, 0, 2)  # [seq_len*field_num, batch_size, emb_size]
        padding_mask = padding_mask.unsqueeze(2)  # [batch_size, seq_len, 1], if treat original missing feature fields as paddings in self-attention, comment this
        padding_mask = padding_mask.repeat(1, 1, len(self.feature_fields)+1)  # [batch_size, seq_len, field_num], if treat original missing feature fields as paddings in self-attention, comment this
        padding_mask = padding_mask.reshape(shape[0], shape[1]*shape[2])  # [batch_size, seq_len*field_num]
        #cross_mask = self.generate_cross_mask(shape[1], shape[2])
        for mod in self.net:
            #temps = mod(temps, src_key_padding_mask=padding_mask, src_mask=cross_mask)  # [seq_len*field_num, batch_size, emb_size]
            temps = mod(temps, src_key_padding_mask=padding_mask)  # [seq_len*field_num, batch_size, emb_size]
        temps = temps.permute(1, 0, 2)  # [batch_size, seq_len*field_num, emb_size]
        temps = temps.reshape(shape[0], shape[1], shape[2], shape[3])  # [batch_size, seq_len, field_num, emb_size]
        temps = temps.permute(2, 0, 1, 3)  # [field_num, batch_size, seq_len, emb_size]
        outputs = []  # [field_num, batch_size, seq_len, *]
        output = torch.einsum('ijk,lk->ijl', temps[0], self.item_embeddings.weight)  # [batch_size, seq_len, item_num]
        outputs.append(output)
        f = 0
        while f < len(self.feature_fields):
            if self.feature_fields[f][1]:
                output = torch.einsum('ijk,lk->ijl', temps[f+1], self.output_mlps[f].weight)  # [batch_size, seq_len, *]
            else:
                output = self.output_mlps[f](temps[f+1])  # [batch_size, seq_len, *]
            outputs.append(output)
            f += 1
        return outputs

    def mii_loss(self, outputs, session_outputs, loss_mask):
        mii_loss = 0
        num = 0  # the number of the masked feature fields in each session
        f = 0
        while f < len(self.feature_fields)+1:
            if f == 0:  # item id
                f_ffd = session_outputs[f]  # [batch_size, seq_len]
                rec_f_ffd = outputs[f]  # [batch_size, seq_len, item_num]
                mask = loss_mask[:,:,f]  # [batch_size, seq_len]
                shape = rec_f_ffd.shape  # we may be not able to use F.cross_entropy when use large batch size (encounter THCudaTensor sizes too large for THCDeviceTensor conversion)
                rec_f_ffd = torch.reshape(rec_f_ffd, (shape[0]*shape[1], shape[2]))  # [batch_size*seq_len, item_num]
                f_ffd = torch.reshape(f_ffd, (shape[0]*shape[1], 1))  # [batch_size*seq_len, 1]
                f_loss = -rec_f_ffd.log_softmax(dim=-1).gather(dim=-1, index=f_ffd).squeeze(-1)  # [batch_size*seq_len]
                f_loss = torch.reshape(f_loss, (shape[0], shape[1]))  # [batch_size, seq_len]
                f_loss = f_loss*mask  # [batch_size, seq_len]
            else:  # other feature fields
                f_ffd = session_outputs[f]  # [batch_size, seq_len, *]
                rec_f_ffd = outputs[f]  # [batch_size, seq_len, *]
                mask = loss_mask[:,:,f]  # [batch_size, seq_len]
                activation = self.feature_fields[f-1][-1]
                if activation == 'sigmoid':
                    norm_rec_f_ffd = torch.sigmoid(rec_f_ffd)
                    f_loss = F.binary_cross_entropy(norm_rec_f_ffd, f_ffd, reduce=False).sum(-1)  # [batch_size, seq_len]
                    f_loss = f_loss*mask  # [batch_size, seq_len]
                elif activation == 'softmax':
                    norm_rec_f_ffd = torch.softmax(rec_f_ffd, -1)
                    f_loss = (f_ffd*norm_rec_f_ffd).sum(-1)  # [batch_size, seq_len]
                    f_loss = f_loss+(f_loss == 0).float()*1e-4  # avoid log0
                    f_loss = -torch.log(f_loss)  # [batch_size, seq_len]
                    f_loss = f_loss*mask  # [batch_size, seq_len]
                else:  # activation == None
                    f_loss = F.mse_loss(rec_f_ffd, f_ffd, reduce=False)  # [batch_size, seq_len, *]
                    f_loss = f_loss.sum(-1)  # [batch_size, seq_len]
                    f_loss = f_loss*mask  # [batch_size, seq_len]
            mii_loss += f_loss.sum(-1)  # [batch_size]
            num += mask.sum(-1)  # [batch_size]
            f += 1
        mii_loss = mii_loss/((num == 0).float()+num)  # [batch_size], if num=0, then loss=0, averaged by the number of the masked feature fields
        return mii_loss

    def rec_loss(self, outputs, output_session_ids, loss_mask):  # because some outputs are not used to calculate the loss, which will lead to an error in DPP
        output = outputs[0]  # [batch_size, seq_len, item_num]
        shape = output.shape  # we may be not able to use F.cross_entropy when use large batch size (encounter THCudaTensor sizes too large for THCDeviceTensor conversion)
        output = torch.reshape(output, (shape[0]*shape[1], shape[2]))  # [batch_size*seq_len, item_num]
        output_session_ids = torch.reshape(output_session_ids, (shape[0]*shape[1], 1))  # [batch_size*seq_len, 1]
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=output_session_ids).squeeze(-1)  # [batch_size*seq_len]
        rec_loss = torch.reshape(rec_loss, (shape[0], shape[1]))  # [batch_size, seq_len]
        rec_loss = rec_loss*loss_mask  # [batch_size, seq_len]
        num = loss_mask.sum(-1)  # [batch_size]
        rec_loss = rec_loss.sum(-1)/((num == 0).float()+num)  # [batch_size]
        return rec_loss
