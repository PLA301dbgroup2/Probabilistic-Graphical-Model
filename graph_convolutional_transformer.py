import random
import numpy as np
import os
import sys
import math
import torch
from torch import nn
from avg_pool import AvgPoolerAggregator
from utils import get_extended_attention_mask
from layers.prompt_layers import *
from layers.poolers import DynamicRoutingAggregator
from layers.myloss import FocalLoss
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Parameter(pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class FeatureEmbedder(nn.Module):
    def __init__(self, args):
        super(FeatureEmbedder, self).__init__()
        self.embeddings = {}
        self.feature_keys = args.feature_keys
        self.dx_embeddings = nn.Embedding(args.vocab_sizes['dx_ints']+1, args.hidden_size, padding_idx=args.vocab_sizes['dx_ints'])
        self.proc_embeddings = nn.Embedding(args.vocab_sizes['proc_ints']+1, args.hidden_size, padding_idx=args.vocab_sizes['proc_ints'])
        self.visit_embeddings = nn.Embedding(1, args.hidden_size)
        self.do_code_prompt = args.do_code_prompt
        self.do_edge_prompt = args.do_edge_prompt
        self.do_label_prompt = args.do_label_prompt
        self.label_key = args.label_key
        self.code_prompt_emmbeddings = CodePrompt(args)
        self.edge_prompt_embeddings = EdgePrompt(args)
        self.label_prompt_embeddings = LabelPrompt(args)
        
        self.hidden_size = args.hidden_size

        self.positional_encoding_dx = PositionalEncoding(self.hidden_size, args.vocab_sizes['dx_ints']+1)
        self.positional_encoding_proc = PositionalEncoding(self.hidden_size, args.vocab_sizes['proc_ints']+1)

        if args.init_by_bert:
            self.map_layer = nn.Linear(768, self.hidden_size)
            self.init_by_bert(args.dim_reduction)

        self.layernorm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
    def forward(self, features):
        batch_size = features[self.feature_keys[0]].shape[0]
        embeddings = {}
        masks = {}
        device = features['dx_ints'].device


        embeddings['dx_ints'] = self.dx_embeddings(features['dx_ints'])
        embeddings['proc_ints'] = self.proc_embeddings(features['proc_ints'])

        embeddings['dx_ints'] = self.positional_encoding_dx(embeddings['dx_ints'].transpose(0, 1)).transpose(0, 1)
        embeddings['proc_ints'] = self.positional_encoding_dx(embeddings['proc_ints'].transpose(0, 1)).transpose(0, 1)

        if self.do_code_prompt:
            embeddings['code_prompts'], masks['code_prompts'] = self.code_prompt_emmbeddings(features['dx_ints'].shape[0])
        
        if self.do_label_prompt:
            embeddings['label_prompts'], masks['label_prompts'] = self.label_prompt_embeddings(features['dx_ints'].shape[0], self.label_key)
        
        embeddings['visit'] = self.visit_embeddings(torch.tensor([0]).to(device))
        embeddings['visit'] = embeddings['visit'].unsqueeze(0).expand(batch_size,-1,-1)
        masks['visit'] = torch.ones(batch_size,1).to(device)
      
        return embeddings, masks
    
    def init_by_bert(self, dim_reduction):
        device = self.dx_embeddings.weight.device
        dx_bert_embeds = torch.load('dataset/dx_embeds.pt').to(device)
        proc_bert_embeds = torch.load('dataset/proc_embeds.pt').to(device)
        dx_num, hidden_size = dx_bert_embeds.shape
        proc_num, _ = proc_bert_embeds.shape
        # assert hidden_size == self.hidden_size, f"embedds initilized by BERT, hidden_size {hidden_size} is required but get {self.hidden_size}"
        if dim_reduction == 'truncate':
            dx_bert_embeds = torch.cat((dx_bert_embeds[:, 0: self.hidden_size], self.dx_embeddings.weight[dx_num:]))
            proc_bert_embeds = torch.cat((proc_bert_embeds[:, 0: self.hidden_size], self.proc_embeddings.weight[proc_num:]))
        else:
            dx_bert_embeds = torch.cat((self.map_layer(dx_bert_embeds), self.dx_embeddings.weight[dx_num:]))
            proc_bert_embeds = torch.cat((self.map_layer(proc_bert_embeds), self.proc_embeddings.weight[proc_num:]))
        self.dx_embeddings.weight = nn.Parameter(dx_bert_embeds)
        self.proc_embeddings.weight = nn.Parameter(proc_bert_embeds)


        
class SelfAttention(nn.Module):
    def __init__(self, args, stack_idx):
        super(SelfAttention, self).__init__()
        self.stack_idx = stack_idx
        self.num_attention_heads = args.num_heads
        self.attention_head_size = int(args.hidden_size / args.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.do_edge_prompt = args.do_edge_prompt
        self.edge_prompt_to_each_stack = args.edge_prompt_to_each_stack
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        # experiment with gate
        self.do_gate_mechanism = args.do_gate_mechanism
        self.pooler = AvgPoolerAggregator()
        # self.pooler = SelfAttnAggregator(args.hidden_size)
        self.gate = nn.Linear(args.hidden_size, 2)
        # experiment with dropout after completion
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True):
        
        if self.do_gate_mechanism and self.do_edge_prompt:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            # take dot product between query and key to get raw attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # prior attention_probs
            if prior is not None:
                attention_probs_prior = prior[:,None,:,:].expand(-1, self.num_attention_heads, -1, -1)
            else:
                attention_probs_prior = torch.zeros_like(attention_probs)


            attention_mask_1 = (attention_mask.squeeze() == 0).float()

            h_ = self.pooler(hidden_states, attention_mask_1)
            gate_scores = self.gate(h_)
            gate_scores = nn.Softmax(dim=-1)(gate_scores / 2)
            bsz = attention_probs.shape[0]
            length_ = attention_probs.shape[2]

            attention_probs = attention_probs * gate_scores[: , 0].view(bsz, 1, 1, 1) \
                            + attention_probs_prior * gate_scores[: , 1].view(bsz, 1, 1, 1)
        elif self.do_edge_prompt:
            edge_prompt = prior[:,None,:,:].expand(-1, self.num_attention_heads, -1, -1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if self.edge_prompt_to_each_stack:
                attention_probs = nn.Softmax(dim=-1)(attention_probs+edge_prompt+ attention_mask)
            else:
               if self.stack_idx == 0:
                   attention_probs = nn.Softmax(dim=-1)(attention_probs+edge_prompt+ attention_mask)
        else:
            if self.stack_idx == 0 and prior is not None:
                attention_probs = prior[:,None,:,:].expand(-1, self.num_attention_heads, -1, -1)
            else:
                mixed_query_layer = self.query(hidden_states)
                mixed_key_layer = self.key(hidden_states)
                query_layer = self.transpose_for_scores(mixed_query_layer)
                key_layer = self.transpose_for_scores(mixed_key_layer)
                # take dot product between query and key to get raw attention scores
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                attention_probs = nn.Softmax(dim=-1)(attention_scores)
            
            
        mixed_value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # dropping out entire tokens to attend to; extra experiment
        # attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs

    
class SelfOutput(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states
        


class Attention(nn.Module):
    def __init__(self, args, stack_idx):
        super(Attention, self).__init__()
        self.self_attention = SelfAttention(args, stack_idx)
        self.self_output = SelfOutput(args)
        
    def forward(self, hidden_states, attention_mask, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.self_attention(hidden_states, attention_mask, guide_mask, prior, output_attentions)
        attention_output = self.self_output(self_attention_outputs[0], hidden_states)
        outputs = (attention_output,) + self_attention_outputs[1:]
        return outputs

class IntermediateLayer(nn.Module):
    def __init__(self, args):
        super(IntermediateLayer, self).__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        self.activation = nn.ReLU()
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
    
class OutputLayer(nn.Module):
    def __init__(self, args):
        super(OutputLayer, self).__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class GCTLayer(nn.Module):
    def __init__(self, args, stack_idx):
        super(GCTLayer, self).__init__()
        self.attention = Attention(args, stack_idx)

    
    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.attention(hidden_states, attention_mask, guide_mask, prior, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        
        outputs = (attention_output,) + outputs
        return outputs

class Pooler(nn.Module):
    def __init__(self, args):
        super(Pooler, self).__init__()
        self.do_label_prompt = args.do_label_prompt

        self.pooler = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU()
        )
        # normalizing
        self.lm = nn.LayerNorm(args.hidden_size)
        
    
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:,0]
        if self.do_label_prompt:
            label_token_tensor = hidden_states[:, -2:]
            return self.lm(first_token_tensor + self.pooler(first_token_tensor)), self.lm(label_token_tensor + self.pooler(label_token_tensor))

        else:
            return self.lm(first_token_tensor + self.pooler(first_token_tensor)), None

class GraphConvolutionalTransformer(nn.Module):
    def __init__(self, args):
        super(GraphConvolutionalTransformer, self).__init__()
        self.num_labels = args.num_labels if not args.mdp else args.vocab_sizes['dx_ints'] + 1
        self.label_key = args.label_key if not args.mdp else 'diagnosis'
        self.reg_coef = args.reg_coef
        self.use_guide = args.use_guide
        self.use_prior = args.use_prior
        self.prior_scalar = args.prior_scalar
        self.batch_size = args.batch_size
        self.num_stacks = args.num_stacks
        self.max_num_codes = args.max_num_codes
        self.output_attentions = args.output_attentions
        self.output_hidden_states = args.output_hidden_states
        self.do_label_prompt = args.do_label_prompt
        self.do_code_prompt = args.do_code_prompt
        self.do_edge_prompt = args.do_edge_prompt
        self.prompt_num = args.prompt_num
        self.feature_keys = args.feature_keys
        self.layers = nn.ModuleList([GCTLayer(args, i) for i in range(args.num_stacks)])
        self.embeddings = FeatureEmbedder(args)
        self.pooler = Pooler(args)
        self.loss_type = 'ce_loss' if not args.focal_loss else 'focal_loss'

        self.use_adr_pooler = args.use_adr_pooler
        self.adr_pooler = DynamicRoutingAggregator(
                args.hidden_size,
                4,
                4,
                iter_num=args.iter_num,
            )

        self.dropout = nn.Dropout(0.2)
        self.expired_classifier = nn.Linear(args.hidden_size, 2)
        self.masked_diagnosis_code_classifier = nn.Linear(args.hidden_size, self.num_labels)
        if args.share_weights:
            self.masked_diagnosis_code_classifier.weight = self.embeddings.dx_embeddings.weight
            

        if self.label_key == 'los':
            self.classifier = nn.Linear(args.hidden_size, 5)
        elif self.label_key == 'expired':
            self.classifier = self.expired_classifier
        elif self.label_key == 'diagnosis':
            self.classifier = self.masked_diagnosis_code_classifier


    
    def get_adj(self, x, y):
        """
        x: batch x x_seq_len x hidden_size
        y: batch x y_seq_len x hidden_size
        return:
        weights_adaptive: batch x seq_len x seq_len 
        
        """

        values = torch.bmm(x, y.transpose(1, 2).contiguous())/math.sqrt(x.shape[-1])
        weights = torch.sigmoid(values)
        
        weights_adaptive = torch.where(weights<0.001, 0, weights)
        
        return weights_adaptive

    def token_view(self, embedding_dict):
        
        keys = embedding_dict.keys()

        
        dp_embeddings = torch.cat((embedding_dict['dx_ints'], embedding_dict['proc_ints']), dim=1)
        if 'code_prompts' in keys  and 'label_prompts' in keys:
            cl_embeddings = torch.cat((embedding_dict['code_prompts'], embedding_dict['label_prompts']), dim=1)
            outer_adj = self.get_adj(dp_embeddings, cl_embeddings)
            inner_adj = self.get_adj(cl_embeddings, cl_embeddings)
        elif 'code_prompts' in keys:
            outer_adj = self.get_adj(dp_embeddings, embedding_dict['code_prompts'])
            inner_adj = self.get_adj(embedding_dict['code_prompts'], embedding_dict['code_prompts'])
        elif 'label_prompts' in keys:
            outer_adj = self.get_adj(dp_embeddings, embedding_dict['label_prompts'])
            inner_adj = self.get_adj(embedding_dict['label_prompts'], embedding_dict['label_prompts'])
        else:
            outer_adj = None
            inner_adj = None
        return outer_adj, inner_adj


    def create_matrix_vdp_0618(self, features, masks, priors, batch_masked_pos=None, outer_adj=None, inner_adj=None):
        """
        Args:

        return:

            prior_guide: 
                batch_size * seq_len * seq_len, seq_len = 1 (visit) + max_dx_num + max_proc_num + prompt_num + 1 (label)
        """
        batch_size = features['dx_ints'].shape[0]
        device = features['dx_ints'].device
        num_dx_ids = self.max_num_codes if self.use_prior else features['dx_ints'].shape[-1]
        num_proc_ids = self.max_num_codes if self.use_prior else features['proc_ints'].shape[-1]
        num_codes = 1 + num_dx_ids + num_proc_ids
        if self.do_code_prompt:
            num_codes += self.prompt_num
        if self.do_label_prompt:
            if self.label_key == 'expired':
                num_abel_prompt = 2
                num_codes += num_abel_prompt
            elif self.label_key == 'los':
                num_abel_prompt = 5
                num_codes += num_abel_prompt
                
        guide = None
        
        if self.use_guide:
            row0 = torch.cat([torch.zeros([1,1]), torch.ones([1, num_dx_ids]), torch.zeros([1,num_proc_ids])], axis=1) 
            row1 = torch.cat([torch.zeros([num_dx_ids,num_dx_ids+1]), torch.ones([num_dx_ids, num_proc_ids])], axis=1) 
            row2 = torch.zeros([num_proc_ids, num_dx_ids+num_proc_ids+1]) 
            guide = torch.cat([row0, row1, row2], axis=0) 
            if self.do_code_prompt:
                row0 = torch.cat([row0, torch.zeros([1, self.prompt_num])], axis=1) 
                row1 = torch.cat([row1, torch.ones([num_dx_ids, self.prompt_num])], axis=1) 
                row2 = torch.cat([row2, torch.ones([num_proc_ids, self.prompt_num])], axis=1) 
                row3 = torch.cat((torch.zeros([self.prompt_num, 1]), torch.ones([self.prompt_num, num_dx_ids+num_proc_ids+self.prompt_num])), axis=1) 
                guide = torch.cat([row0, row1, row2, row3], axis=0) 
            if self.do_label_prompt:
                row0 = torch.cat([row0 , torch.zeros([1, num_abel_prompt])], axis=1) 
                row1 = torch.cat([row1, torch.ones([num_dx_ids, num_abel_prompt])], axis=1) 
                row2 = torch.cat([row2, torch.ones([num_proc_ids, num_abel_prompt])], axis=1) 
                if self.do_code_prompt:
                    row3 = torch.cat([row3, torch.zeros([self.prompt_num, num_abel_prompt])], axis=1) 
                    row4 = torch.cat((torch.zeros([num_abel_prompt, 1]), torch.ones([num_abel_prompt, num_codes-num_abel_prompt-1-self.prompt_num]), torch.zeros([num_abel_prompt, num_abel_prompt + self.prompt_num])), axis=1)
                    guide = torch.cat([row0, row1, row2, row3, row4], axis=0)
                else:
                    row3 = torch.cat((torch.zeros([2, 1]), torch.ones([2, num_codes-3]), torch.zeros([2, 2])), axis=1)
                    guide = torch.cat([row0, row1, row2, row3], axis=0)
                
            
            guide = guide + guide.t()
            guide = guide.to(device)
            
            guide = guide.unsqueeze(0)
            guide = guide.expand(batch_size, -1, -1)

            guide = (guide*masks.unsqueeze(-1)*masks.unsqueeze(1)+torch.eye(num_codes).to(device).unsqueeze(0))


        if self.use_prior:
            prior_idx = priors['indices'].t() # num * 3 (batch_id, row, col)
            temp_idx = (prior_idx[:,0]*100000 + prior_idx[:,1]*1000 + prior_idx[:,2])
            sorted_idx = torch.argsort(temp_idx)
            prior_idx = prior_idx[sorted_idx]
            
            prior_idx_shape = [batch_size, self.max_num_codes*2, self.max_num_codes*2]
            sparse_prior = torch.sparse_coo_tensor(prior_idx.t(), priors['values'], torch.Size(prior_idx_shape))
            prior_guide = sparse_prior.to_dense() # batch_size x (max_num_codes*2) x (max_num_codes*2)

            if batch_masked_pos is not None:

                for ii, pos in enumerate(batch_masked_pos):
                    prior_guide[ii][self.max_num_codes: , pos] = torch.tensor(0., device=device)
                    prior_guide[ii][pos, self.max_num_codes: ] = torch.tensor(0., device=device)
            num =  num_codes - 1 - num_dx_ids - num_proc_ids
            visit_guide = torch.tensor([self.prior_scalar]*self.max_num_codes + [0.0]*self.max_num_codes*1, dtype=torch.float,device=device) 
            prior_guide = torch.cat([visit_guide.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1), prior_guide], axis=1) 
            visit_guide = torch.cat([torch.tensor([0.0], device=device), visit_guide], axis=0)
            prior_guide = torch.cat([visit_guide.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1), prior_guide], axis=2) 
            if num != 0:
                prompt_guide = torch.cat((torch.zeros([batch_size, num, 1], device=device), outer_adj.transpose(1, 2).contiguous()), axis=2) 
                prior_guide = torch.cat([prior_guide, prompt_guide], axis=1) 
                prompt_guide = torch.cat([prompt_guide.transpose(1, 2).contiguous(), inner_adj], axis=1) 
                prior_guide = torch.cat([prior_guide, prompt_guide], axis=2)

            if guide is not None:
                prior_guide = prior_guide * guide
            prior_guide = (prior_guide*masks.unsqueeze(-1)*masks.unsqueeze(1) + self.prior_scalar*torch.eye(num_codes, device=device).unsqueeze(0))
            degrees = torch.sum(prior_guide, axis=2)
            prior_guide = prior_guide / degrees.unsqueeze(-1)

        return guide, prior_guide
    
    def get_loss(self, logits, labels, attentions):

        loss_fct = nn.CrossEntropyLoss() if self.loss_type == 'ce_loss' else FocalLoss()

        batch_size = labels.shape[0]
        loss = loss_fct(logits.view(batch_size, -1), labels.view(-1))
        
        if self.use_prior:
            kl_terms = []
            for i in range(1, self.num_stacks):
                log_p = torch.log(attentions[i-1] + 1e-12)
                log_q = torch.log(attentions[i] + 1e-12)
                kl_term = attentions[i-1] * (log_p - log_q)
                kl_term = torch.sum(kl_term, axis=-1)
                kl_term = torch.mean(kl_term)
                kl_terms.append(kl_term)
            reg_term = torch.mean(torch.tensor(kl_terms))
            loss += self.reg_coef * reg_term
        return loss


    def forward(self, data, all_priors):

        embedding_dict, mask_dict = self.embeddings(data)

        priors_data, bert_priors_data, kg_priors_data = all_priors
        

        mask_dict['dx_ints'] = data['dx_masks']
        mask_dict['proc_ints'] = data['proc_masks']

        
        keys = ['visit', 'dx_ints', 'proc_ints']
        if self.do_code_prompt: keys.append('code_prompts')
        if self.do_label_prompt: keys.append('label_prompts')
        hidden_states = torch.cat([embedding_dict[key] for key in keys], axis=1)


        masks = torch.cat([mask_dict[key] for key in keys], axis=1)


        outer_adj, inner_adj = self.token_view(embedding_dict)

        guide, prior_guide = self.create_matrix_vdp_0618(data, masks, priors_data, outer_adj=outer_adj, inner_adj=inner_adj)
        bert_guide, bert_prior_guide = self.create_matrix_vdp_0618(data, masks, bert_priors_data, outer_adj=outer_adj, inner_adj=inner_adj)
        kg_guide, kg_prior_guide = self.create_matrix_vdp_0618(data, masks, kg_priors_data, outer_adj=outer_adj, inner_adj=inner_adj)
        
        all_hidden_states = () if self.output_hidden_states else None
        all_attentions = () if self.output_attentions else None

        extended_attention_mask = get_extended_attention_mask(masks)

        extended_guide_mask = get_extended_attention_mask(guide) if self.use_guide else None
        
        
        for i, layer_module in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer_module(hidden_states, extended_attention_mask, extended_guide_mask, prior_guide, self.output_attentions)
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        visit_hidden_state, label_hidden_state = self.pooler(hidden_states) # batch * dim

        if self.use_adr_pooler:
            adr_masks = masks
            if 'label_prompts' in keys:
                adr_masks = torch.cat([mask_dict[key] for key in keys[:-1]], axis=1)
                adr_masks = torch.cat([adr_masks, torch.zeros_like(mask_dict['label_prompts']).to(adr_masks.device)], axis=1)
            visit_hidden_state = self.adr_pooler(hidden_states, adr_masks)

        logits = self.classifier(self.dropout(visit_hidden_state))
        loss = self.get_loss(logits, data[self.label_key], all_attentions)
        
        
        return tuple(v for v in [loss, logits, all_hidden_states, all_attentions] if v is not None)
        