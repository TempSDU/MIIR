import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from contextlib import suppress as nullcontext
from DataLoader import TrainDataset_MII
from Utility import init_seeds, xavier_init
from MIIR import *


def pretrainfunction(accelerator, model, dataloader, optimizer, n_gpu, K, epoch, save_path, log_every_n_steps, accumulate_grad_batches, gradient_clip_val):
    batch_size = dataloader.batch_size
    ddp_model, ddp_dataloader, ddp_optimizer = accelerator.prepare(model, dataloader, optimizer)
    ddp_model.train()
    ddp_optimizer.zero_grad()
    acc_loss = 0
    acc_num = 0
    start = time.time()
    step = 0
    for _ in range(K):
        for batch in ddp_dataloader:
            step += 1
            if step%accumulate_grad_batches == 0 or step == len(ddp_dataloader)//batch_size:
                index, session_input, session_output, session_masks = batch
                output = ddp_model(session_input[0], session_input[1:], session_masks[0])
                if n_gpu > 1:
                    loss = ddp_model.module.mii_loss(output, session_output, session_masks[1])
                else:
                    loss = ddp_model.mii_loss(output, session_output, session_masks[1])
                acc_loss += loss.sum().detach()
                loss = loss.sum()/(batch_size/n_gpu*accumulate_grad_batches)  # one batch data will be split by n_gpu, so in one device, the actual batch size after accumulating gradients will be batch_size/n_gpu*accumulate_grad_batches
                accelerator.backward(loss)
                if gradient_clip_val > 0:
                    accelerator.clip_grad_value_(ddp_model.parameters(), gradient_clip_val)
                ddp_optimizer.step()  # the gradients from all devices will be summed then averaged by n_gpu
                ddp_optimizer.zero_grad()
            else:
                if n_gpu > 1:
                    my_context = ddp_model.no_sync()
                else:
                    my_context = nullcontext()
                with my_context:
                    index, session_input, session_output, session_masks = batch
                    output = ddp_model(session_input[0], session_input[1:], session_masks[0])
                    if n_gpu > 1:
                        loss = ddp_model.module.mii_loss(output, session_output, session_masks[1])
                    else:
                        loss = ddp_model.mii_loss(output, session_output, session_masks[1])
                    acc_loss += loss.sum().detach()
                    loss = loss.sum()/(batch_size/n_gpu*accumulate_grad_batches)  # one batch data will be split by n_gpu, so in one device, the actual batch size after accumulating gradients will be batch_size/n_gpu*accumulate_grad_batches
                    accelerator.backward(loss)
            acc_num += batch_size/n_gpu  # here one batch data will be split by n_gpu
            if torch.isnan(acc_loss):
                if accelerator.is_main_process:
                    print('Epoch %d Step %d Loss NaN Error!' % (epoch, step))
                exit()
            if step%log_every_n_steps == 0:
                accelerator.wait_for_everyone()
                all_acc_loss = accelerator.gather((acc_loss/acc_num).unsqueeze(-1))
                if accelerator.is_main_process:
                    print('Epoch %d Step %d Loss: %0.4f, Time %d' % (epoch, step, (all_acc_loss/n_gpu).sum(), time.time()-start))
    accelerator.wait_for_everyone()
    ddp_model = accelerator.unwrap_model(ddp_model)
    ddp_model.cpu()
    accelerator.save(ddp_model.state_dict(), save_path+'pretrain-'+str(epoch)+'.pth')
    all_acc_loss = accelerator.gather((acc_loss/acc_num).unsqueeze(-1))
    all_acc_loss = (all_acc_loss/n_gpu).sum()
    return all_acc_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_files', type=str, help='the files for a dataset')
    parser.add_argument('-save_path', type=str, help='the path for saving the model')
    parser.add_argument('-n_gpu', type=int, default=1, help='the number of gpus, should be consistent with the num_processes in accelerate_config.yaml')
    parser.add_argument('-K', type=int, default=1, help='the number of iterations in one epoch')
    parser.add_argument('-epochs', type=int, default=100, help='the number of epochs')
    parser.add_argument('-log_every_n_steps', type=int, default=50, help='how often to log within steps')
    parser.add_argument('-accumulate_grad_batches', type=int, default=1, help='the number of batches for accumulating gradients, 1 means don\'t accumulate gradients')
    parser.add_argument('-gradient_clip_val', type=float, default=5, help='the value for clipping gradients, 0 means don\'t clip gradients')
    parser.add_argument('-max_interactions', type=int, default=20, help='the maximum number of the interactions in one input set')
    parser.add_argument('-mask_prob', type=float, default=0.5, help='the probability of a feature field being masked')
    parser.add_argument('-batch_size', type=int, default=128, help='the size of one batch')
    parser.add_argument('-num_workers', type=int, default=10, help='how many subprocesses to use for data loading')
    parser.add_argument('-lr', type=float, default=0.0001, help='the initial learning rate')
    parser.add_argument('-eta_min', type=float, default=0.000001, help='the minimum learning rate')
    parser.add_argument('-T_max', type=int, default=5, help='maximum number of iterations')
    parser.add_argument('-resume', type=bool, default=False, help='resume or not')
    args = parser.parse_args()
    with open(args.dataset_files, 'r') as f:
        content = f.readlines()
    dataset_files = eval(content[0].strip())

    init_seeds()
    accelerator = Accelerator(split_batches=True)
    if accelerator.is_main_process:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if args.resume:
            fw = open(args.save_path+'pretrain_result.txt', 'a')
        else:
            fw = open(args.save_path+'pretrain_result.txt', 'w')
    accelerator.wait_for_everyone()

    last_epoch = 0
    if args.resume:
        with open(args.save_path+'pretrain_result.txt', 'r') as f:
            content = f.readlines()
        last_epoch = len(content)
        model = MIIR(dataset_files['dataset_token'])
        accelerator.print('Load the model from Epoch %d' % (last_epoch,))
        model.load_state_dict(torch.load(args.save_path+'pretrain-'+str(last_epoch)+'.pth'))
    else:
        model = MIIR(dataset_files['dataset_token'])
        accelerator.print('Initialize the model')
        model.apply(xavier_init)
        accelerator.wait_for_everyone()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.eta_min, T_max=args.T_max)
    dataset = TrainDataset_MII(dataset_files['interactions_filepath'], dataset_files['categories_filepath'], dataset_files['brands_filepath'], dataset_files['titles_filepath'], dataset_files['descriptions_filepath'], dataset_files['missings_filepath'], dataset_files['train_rows_filepath'], args.max_interactions, args.mask_prob, dataset_files['masks_filepath'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    epoch = last_epoch
    while epoch < args.epochs:
        start = time.time()
        result = {}
        loss = pretrainfunction(accelerator, model, dataloader, optimizer, args.n_gpu, args.K, epoch+1, args.save_path, args.log_every_n_steps, args.accumulate_grad_batches, args.gradient_clip_val)
        result['epoch'] = epoch+1
        result['loss'] = loss
        if accelerator.is_main_process:
            print('Epoch %d Loss: %0.4f, Time %d' % (epoch+1, loss, time.time()-start))
            fw.write(str(result)+'\n')
            fw.flush()
        #scheduler.step()
        epoch += 1
    if accelerator.is_main_process:
        fw.close()
