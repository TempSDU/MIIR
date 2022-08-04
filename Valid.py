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
from DataLoader import TestDataset_MLM, TestDataset_CLM
from Utility import init_seeds, xavier_init, evaluatefunction
from MIIR import *


def validfunction(accelerator, model, dataloader, epoch, rank, save_path):
    results = []
    indices = []
    ddp_model, ddp_dataloader = accelerator.prepare(model, dataloader)
    ddp_model.eval()
    with torch.no_grad():
        for batch in ddp_dataloader:
            index, session_ids, session_side_information, session_masks, session_negatives = batch
            output = output[0]  # item id
            index = index.cpu().numpy()  # [batch_size]
            output = output[session_masks[1] == 1].cpu().numpy()  # [batch_size, item_num], note that include padding and missing
            session_positives = session_ids[1][session_masks[1] == 1].cpu().numpy()  # [batch_size]
            session_negatives = session_negatives.cpu().numpy()
            result = evaluatefunction(output, session_positives, session_negatives)
            results.extend(result)
            indices.extend(index.tolist())
    with open(save_path+'valid-'+str(epoch)+'-'+str(rank)+'.txt', 'w') as f:
        i = 0
        for result in results:
            result['index'] = indices[i]
            f.write(str(result)+'\n')
            i += 1
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_files', type=str, help='the files for a dataset')
    parser.add_argument('-save_path', type=str, help='the path for saving the model')
    parser.add_argument('-n_gpu', type=int, default=1, help='the number of gpus, should be consistent with the num_processes in accelerate_config.yaml')
    parser.add_argument('-max_interactions', type=int, default=20, help='the maximum number of the interactions in one input set')
    parser.add_argument('-batch_size', type=int, default=256, help='the size of one batch')
    parser.add_argument('-num_workers', type=int, default=10, help='how many subprocesses to use for data loading')
    parser.add_argument('-resume', type=bool, default=False, help='resume or not')
    args = parser.parse_args()
    with open(args.dataset_files, 'r') as f:
        content = f.readlines()
    dataset_files = eval(content[0].strip())
    with open(args.save_path+'train_result.txt', 'r') as f:
        content = f.readlines()
    epochs = len(content)

    init_seeds()
    accelerator = Accelerator(split_batches=True)
    if accelerator.is_main_process:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if args.resume:
            fw = open(args.save_path+'valid_result.txt', 'a')
        else:
            fw = open(args.save_path+'valid_result.txt', 'w')
    accelerator.wait_for_everyone()

    accelerator.print('Evaluate the trained model')
    model = MIIR(dataset_files['dataset_token'])
    dataset = TestDataset_MLM(dataset_files['interactions_filepath'], dataset_files['categories_filepath'], dataset_files['brands_filepath'], dataset_files['titles_filepath'], dataset_files['descriptions_filepath'], dataset_files['missings_filepath'], dataset_files['valid_rows_filepath'], dataset_files['valid_negatives_filepath'], args.max_interactions, dataset_files['masks_filepath'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)  # because drop_last=False, it may be some duplicate instances in the last batch
    next_epoch = 1
    if args.resume:
        with open(args.save_path+'valid_result.txt', 'r') as f:
            content = f.readlines()
        next_epoch = len(content)+1
        accelerator.print('Resume from Epoch %d' % (next_epoch,))

    epoch = next_epoch
    while epoch <= epochs:
        model.load_state_dict(torch.load(args.save_path+'train-'+str(epoch)+'.pth'))
        start = time.time()
        validfunction(accelerator, model, dataloader, epoch, accelerator.process_index, args.save_path)
        if accelerator.is_main_process:  # merge results from all processes
            results = {}
            for rank in range(args.n_gpu):
                with open(args.save_path+'valid-'+str(epoch)+'-'+str(rank)+'.txt', 'r') as f:
                    content = f.readlines()
                for line in content:
                    line = eval(line.strip())
                    index = line['index']
                    results[index] = line
                os.remove(args.save_path+'valid-'+str(epoch)+'-'+str(rank)+'.txt')
            assert len(results) == len(dataset), 'len(results) should be equal to len(dataset)'
            average_result = {}
            with open(args.save_path+'valid-'+str(epoch)+'.txt', 'w') as f:
                for index in range(len(results)):
                    if index == 0:
                        for metric in results[index]:
                            if metric != 'index' and metric != 'predict':
                                average_result[metric] = results[index][metric]
                    else:
                        for metric in results[index]:
                            if metric != 'index' and metric != 'predict':
                                average_result[metric] += results[index][metric]
                    f.write(str(results[index])+'\n')
            metric_sum = 0
            for metric in average_result:
                average_result[metric] /= len(results)
                metric_sum += average_result[metric]
            average_result['sum'] = metric_sum
            print('Epoch %d Time %d:' % (epoch,time.time()-start), average_result)
            average_result['epoch'] = epoch
            fw.write(str(average_result)+'\n')
            fw.flush()
        accelerator.wait_for_everyone()
        epoch += 1
    if accelerator.is_main_process:
        fw.close()
