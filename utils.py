import numpy as np
import os
import sys
import math
import torch
import json
import random
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import argparse

from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    average_precision_score, 
    auc,
    roc_curve,
    precision_recall_curve,
    accuracy_score)

class eICUDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def priors_collate_fn(batch):

    new_batch = []
    for i, item in enumerate(batch):
        num_indices = item[0].shape[-1]
        new_indices = torch.cat((torch.tensor([i]*num_indices).reshape(1,-1), item[0]), axis=0)
        new_batch.append((new_indices, item[1]))
    indices = torch.cat([t[0] for t in new_batch], axis=1)
    values = torch.cat([t[1] for t in new_batch], axis=-1)
    return indices, values

def data_collate_fn_for_mdp(batch):
    """

    Args:
        batch: batch_size x 6 x hidden_size
        6: dx_ints, proc_ints, dx_masks, proc_masks, readmission_labels, expired_labels
    return: 
        batch, label, masked_position
    """

    new_batch = []
    for ii, item in enumerate(batch):
        num_dx = torch.sum(item[2]) 
        assert num_dx!=0, item[2]
        masked_pos = random.randint(0, num_dx-1) #

        new_dx_ints = item[0].clone()
        new_dx_ints[masked_pos] = torch.tensor([3862]) 
        new_batch.append((new_dx_ints, item[1], item[2], item[3], item[4], item[5], item[0][masked_pos]))

        
        

    dx_ints = torch.stack([t[0] for t in new_batch], axis=0)
    proc_ints = torch.stack([t[1] for t in new_batch], axis=0)
    dx_masks = torch.stack([t[2] for t in new_batch], axis=0)
    proc_masks = torch.stack([t[3] for t in new_batch], axis=0)
    readmission_labels = torch.stack([t[4] for t in new_batch], axis=0)
    expired_labels = torch.stack([t[5] for t in new_batch], axis=0)
    #los_labels = torch.stack([t[6] for t in new_batch], axis=0)
    diagnosis_labels = torch.stack([t[6] for t in new_batch], axis=0)

    return dx_ints, proc_ints, dx_masks, proc_masks, readmission_labels, expired_labels, diagnosis_labels, torch.tensor(masked_pos)


def data_collate_fn_for_los(batch):
    """
    """
    dx_ints = torch.stack([t[0] for t in batch], axis=0)
    proc_ints = torch.stack([t[1] for t in batch], axis=0)
    dx_masks = torch.stack([t[2] for t in batch], axis=0)
    proc_masks = torch.stack([t[3] for t in batch], axis=0)
    readmission_labels = torch.stack([t[4] for t in batch], axis=0)
    expired_labels = torch.stack([t[5] for t in batch], axis=0)

    
    def los_map_to_label(x):
        if x <= 3 and x >=0:
            return torch.tensor(0)
        elif x <= 7 and x > 3:
            return torch.tensor(1)
        elif x <= 14 and x > 7:
            return torch.tensor(2)
        elif x <= 30 and x > 14:
            return torch.tensor(3)
        elif x > 30:
            return torch.tensor(4)
        else:
            x = 0
            return torch.tensor(0)


    los_labels = torch.stack([los_map_to_label(t[6]) for t in batch], axis=0)

    return dx_ints, proc_ints, dx_masks, proc_masks, readmission_labels, expired_labels, los_labels

    
def get_extended_attention_mask(attention_mask):
    if attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps-num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_exponential_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    alpha = 0.0002
    beta = 0.0002
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0001, math.exp(-beta * float(current_step-num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    T_max = (num_training_steps-num_warmup_steps) // 1
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return (math.cos(max(0.0, math.pi * float(current_step-num_warmup_steps) / T_max)) + 1) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def nested_concat(tensors, new_tensors, dim=0):
    "Concat the `new_tensors` to `tensors` on `dim`. Works for tensors or nested list/tuples of tensors."
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def prepare_data(data, priors_data, device):
    
    features = {}
    if len(data) == 8: # for diagnosis prediction
        feat_keys = ['dx_ints', 'proc_ints', 'dx_masks', 'proc_masks', 'readmission', 'expired', 'diagnosis', 'masked_pos_in_dx']
    elif len(data) == 7: # for length of stay prediction
        feat_keys = ['dx_ints', 'proc_ints', 'dx_masks', 'proc_masks', 'readmission', 'expired', 'los']
    elif len(data) == 6: # for expired rate prediction
        feat_keys = ['dx_ints', 'proc_ints', 'dx_masks', 'proc_masks', 'readmission', 'expired']
    for ii, item in enumerate(data):
        features[feat_keys[ii]] = item

    for k, v in features.items():
        features[k] = v.to(device)
    priors = {}
    priors['indices'] = priors_data[0].to(device)
    priors['values'] = priors_data[1].to(device)
        
    
    return features, priors
    
def compute_metrics(preds, labels):
    metrics = {}
    shape = preds.shape
    preds = np.argmax(preds, axis=1)
    if shape[1] > 2:
        metrics['ACC'] = accuracy_score(labels, preds)
    else:
        # average precision
        ap = average_precision_score(labels, preds)
        precisions, recalls, thresholds = precision_recall_curve(labels, preds)
        auc_pr = auc(recalls, precisions)
        metrics['AUCPR'] = auc_pr

        
        
        # auroc
        fpr, tpr, thresholds = roc_curve(labels, preds)
        auc_roc = auc(fpr, tpr)
        metrics['AUROC'] = auc_roc
        
        # f1 score, precision, recall
        precision, recall, fscore, support = precision_recall_fscore_support(labels, preds, average='weighted')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['fscore'] = fscore
    
    return metrics
"""
for name, param in model.named_parameters():
    print(name, param.is_cuda)
"""
class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--data_dir', type=str, required=True)
        self.add_argument('--output_dir', type=str, required=True)

        self.add_argument('--max_num_codes', type=int, default=200)
        self.add_argument('--feature_keys', action='append', default=['dx_ints','proc_ints'])
      

        self.add_argument('--vocab_sizes', type=json.loads, default={'dx_ints':3862, 'proc_ints':2709})

        self.add_argument('--prior_scalar', type=float, default=0.5)
        
        self.add_argument('--num_stacks', type=int, required=True)
        self.add_argument('--hidden_size', type=int, default=128) # 128
        self.add_argument('--intermediate_size', type=int, default=256) # 256
        self.add_argument('--num_heads', type=int, default=2)
        self.add_argument('--hidden_dropout_prob', type=float, default=0.25)

        
        self.add_argument('--learning_rate', type=float, default=1e-3)
        self.add_argument('--eps', type=float, default=1e-8)
        self.add_argument('--batch_size', type=int, default=256)
        self.add_argument('--max_grad_norm', type=float, default=1.0)
        
        self.add_argument('--use_guide', default=True, action='store_true')
        self.add_argument('--use_prior', default=True, action='store_true')

        self.add_argument('--output_hidden_states', default=False, action='store_true')
        self.add_argument('--output_attentions', default=False, action='store_true')
        
        self.add_argument('--fold', type=int, default=42)
        self.add_argument('--eval_batch_size', type=int, default=512)
        
        self.add_argument('--warmup', type=float, default=0.1)
        self.add_argument('--logging_steps', type=int, default=200)
        self.add_argument('--max_steps', type=int, default=31250)
        self.add_argument('--num_train_epochs', type=int, default=0)
        
        self.add_argument('--label_key', type=str, default='expired', choices=['readmission', 'expired', 'diagnosis', 'los'])
        self.add_argument('--num_labels', type=int, default=2)
        
        self.add_argument('--reg_coef', type=float, default=0)
        self.add_argument('--seed', type=int, default=42)
        
        
        self.add_argument('--do_train', default=False, action='store_true')
        self.add_argument('--do_eval', default=False, action='store_true')
        self.add_argument('--do_test', default=False, action='store_true')


        self.add_argument('--init_by_bert', default=False, action='store_true')
        self.add_argument('--prior_type', type=str, default='bert_prior', choices=['bert_prior', 'co_occur', 'co_and_kg'])
        self.add_argument('--share_weights', default=True, action='store_true')
        self.add_argument('--mdp', default=False, action='store_true', help='boolean type, true for masked diagnosis prediction, fasle for expired prediction')
        self.add_argument('--do_code_prompt', default=False, action='store_true')
        self.add_argument('--do_edge_prompt', default=False, action='store_true')
        self.add_argument('--do_label_prompt', default=False, action='store_true')
        self.add_argument('--prompt_num', type=int, default=3)
        self.add_argument('--init_label_prompt_by_bert', default=False, action='store_true')
        self.add_argument('--dim_reduction', type=str, default='truncate', choices=['truncate', 'linaer_trans'],
                          help='used for reduce dimension of label embedding outputed from bert')
        self.add_argument('--edge_prompt_to_each_stack', default=True, action='store_true',
                          help='true for adding edge prompt to each stack of attention_probs, false for adding edge prompt to first stack of attention_probs')
        self.add_argument('--do_gate_mechanism', default=False, action='store_true',
                          help='true for use gate mechanism')
        self.add_argument('--use_adr_pooler', default=False, action='store_true',
                          help='true for use AvgPoolerAggregator instead of visit representation')
        self.add_argument('--iter_num', default=3, type=int)
        self.add_argument('--log_dir', default='dirs_${LR}_${DROPOUT}/logging/training.log', type=str)
        self.add_argument('--task_desc', type=str, default='')
        self.add_argument('--mdp_v1', default=False, action='store_true')
        self.add_argument('--device', type=lambda x: f"cuda:{x}", default='cpu')
        self.add_argument('--do_prompt', default=False, action='store_true', help='True for do prompt for expired/readmission task')
        self.add_argument('--focal_loss', default=False, action='store_true', help='true for use focal loss false for not')
        self.add_argument('--save_model_path', default='saved_models/')
        self.add_argument('--early_stop_time', default=50, type=int, help='training will be stoped until metric never promote to early_stop_time')
        self.add_argument('--fine_tune', default=False, action='store_true', help='true for fine tuning prompt parameters and dense layer parameters')
        
        

    def parse_args(self):
        args = super().parse_args()
        return args