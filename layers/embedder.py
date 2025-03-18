import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForNextSentencePrediction
import pickle
from tqdm import tqdm
import os
import argparse
import gc 

class PackedBert(nn.Module):

    def __init__(self, model_path ='bert_path', map_path='dataset/hf0812', pooling='cls', dropout=0.2, device='cpu'):
        super(PackedBert, self).__init__()
        config = AutoConfig.from_pretrained(model_path)
       
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        config.output_hidden_states = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embedding_bert = AutoModel.from_pretrained(model_path)
        self.embedding_bert.to(device)
        self.device = device
        self.bert = BertForNextSentencePrediction.from_pretrained(model_path, output_hidden_states = True)
        self.bert.to(device)
        self.pooling = pooling
        self.dx_int2str = pickle.load(open(os.path.join(map_path, 'dx_int2str.p'), 'rb'))
        self.proc_int2str = pickle.load(open(os.path.join(map_path, 'treat_int2str.p'), 'rb'))
        self.map_path = map_path
        self.dx_map = pickle.load(open(os.path.join(map_path, 'dx_map.p'), 'rb'))
        self.proc_map = pickle.load(open(os.path.join(map_path, 'proc_map.p'), 'rb'))
       
    def get_seq_embedding(self, seq):
        device = self.device
        token = self.tokenizer(seq)

        out = self.embedding_bert(torch.tensor([token.token_type_ids]).to(device), output_hidden_states=True, return_dict=True)

        if self.pooling == 'cls':
            return (out.last_hidden_state[:, 0]).to('cpu')  # [batch, 768]


        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]


        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

    def code2embeds(self):

        dx_embeds = None
        proc_embeds = None
        self.bert.eval()
        for key in tqdm(sorted(self.dx_map.items(), key=lambda x: x[1])):
            dx, dx_id = key
            dx_embeds = (self.get_seq_embedding(dx)).cpu() if dx_embeds is None else torch.cat((dx_embeds, (self.get_seq_embedding(dx)).cpu()), dim=0)
            torch.cuda.empty_cache()
        torch.save(dx_embeds, os.path.join(self.map_path, 'dx_embeds.pt'))
        for key in tqdm(sorted(self.proc_map.items(), key=lambda x: x[1])):
            proc, proc_id = key
            proc_embeds = (self.get_seq_embedding(proc)).cpu() if proc_embeds is None else torch.cat((proc_embeds, (self.get_seq_embedding(proc)).cpu()), dim=0)
        torch.save(proc_embeds, os.path.join(self.map_path, 'proc_embeds.pt'))
    
    def get_nsp_probs(self, text1, text2, batch_size=80):
        
        device = self.device
        assert len(text1)>batch_size, f'batch_size {batch_size} is larger than length of text list {len(text1)}'

        indices = [i*batch_size for i in range(len(text1)//batch_size + 1)] + [len(text1)]
        dp_cond_probs = {}
        pd_cond_probs = {}
        dp_duplicated = 0
        pd_duplicated = 0


        softmax = torch.nn.Softmax(dim=1)


        with torch.no_grad(): 

            for ii, jj in tqdm(zip(indices[:-1], indices[1:])):
                b_text1, b_text2 = text1[ii:jj], text2[ii:jj]
                tokens = self.tokenizer(b_text1, b_text2, padding=True)
                b_input_ids, b_token_type_ids, b_attention_mask = tokens.input_ids, tokens.token_type_ids, tokens.attention_mask
                b_input_ids_tensor = torch.tensor(b_input_ids)
                b_token_type_ids_tensor = torch.tensor(b_token_type_ids)
                b_attention_mask_tensor = torch.tensor(b_attention_mask)
                prediction=self.bert(b_input_ids_tensor.to(device), token_type_ids=b_token_type_ids_tensor.to(device), attention_mask=b_attention_mask_tensor.to(device))
                prediction_sm = softmax(prediction.logits)
                
                for para in [tokens, b_input_ids, b_token_type_ids, b_attention_mask, b_input_ids_tensor, b_token_type_ids_tensor, b_attention_mask_tensor, prediction, prediction_sm]:
                    del para
                for dx, proc, prob in zip(b_text1, b_text2, prediction_sm.to('cpu').tolist()):
                    dp = dx+','+proc
                    if dp in dp_cond_probs.keys():
                        dp_duplicated += 1
                    else:
                        dp_cond_probs[dp] = prob[0]

                    del dx, proc, prob

            print(f'dp_duplicated of dp_probs is {dp_duplicated}')
            print(f'length of dp_cond_probs is {len(dp_cond_probs)}')
            
            for ii, jj in tqdm(zip(indices[:-1], indices[1:])):
                b_text1, b_text2 = text2[ii:jj], text1[ii:jj]
                tokens = self.tokenizer(b_text1, b_text2, padding=True)
                b_input_ids, b_token_type_ids, b_attention_mask = tokens.input_ids, tokens.token_type_ids, tokens.attention_mask
                b_input_ids_tensor = torch.tensor(b_input_ids)
                b_token_type_ids_tensor = torch.tensor(b_token_type_ids)
                b_attention_mask_tensor = torch.tensor(b_attention_mask)
                prediction=self.bert(b_input_ids_tensor.to(device), token_type_ids=b_token_type_ids_tensor.to(device), attention_mask=b_attention_mask_tensor.to(device))
                prediction_sm = softmax(prediction.logits)

                for para in [tokens, b_input_ids, b_token_type_ids, b_attention_mask, b_input_ids_tensor, b_token_type_ids_tensor, b_attention_mask_tensor, prediction, prediction_sm]:
                    del para
                for proc, dx, prob in zip(b_text1, b_text2, prediction_sm.to('cpu').tolist()):
                    pd = proc+','+dx
                    if pd in pd_cond_probs.keys():
                        pd_duplicated += 1
                    else:
                        pd_cond_probs[pd] = prob[0] 

                    del proc, dx, prob   
        print(f'number of pd_duplicated is {pd_duplicated}')
        print(f'length of pd_cond_probs is {len(pd_cond_probs)}')
        return dp_cond_probs, pd_cond_probs

    def count_dp_cond_probs(self, enc_features_list, output_path, train_key_set=None):

        device = self.device
        dx_idx = []
        proc_idx = []


        for enc_feature in tqdm(enc_features_list):
            key = enc_feature.patient_id
            if (train_key_set is not None and key not in train_key_set):
                continue
            dx_ids = enc_feature.dx_ids
            proc_ids = enc_feature.proc_ids

            for dx in dx_ids:
                for proc in proc_ids:
                    dx_idx.append(dx)
                    proc_idx.append(proc)

        
        batch_size = 10000 

        dx_idx_batches = [dx_idx[i:i+batch_size] for i in range(0, len(dx_idx), batch_size)]
        

        dx_idx_set = set()
        for batch in tqdm(dx_idx_batches, desc="Processing batches"):
            dx_idx_set.update(batch)
            del batch


        
        dx_idx_list = []
        proc_idx_list = []


        for dp_id in tqdm(dx_idx_set, desc="Processing dx_idx_set"):
            for pd_id in set(proc_idx):
                dx_idx_list.append(dp_id)
                proc_idx_list.append(pd_id)
            del dp_id, pd_id  

        
        print(f'length of dx_idx is {len(dx_idx_set)}')
        print(f'length of proc_idx is {len(set(proc_idx))}')
        print(f'length of proc_idx_list is {len(proc_idx_list)}')
        print(f'length of unique proc_idx_list is {len(set(proc_idx_list))}')
        
        dp_cond_probs, pd_cond_probs = self.get_nsp_probs(dx_idx_list, proc_idx_list)


        try:

            os.makedirs(output_path, exist_ok=True)
            

            with open(os.path.join(output_path, 'dp_cond_probs.nsp.p'), 'wb') as f:
                pickle.dump(dp_cond_probs, f)
            

            with open(os.path.join(output_path, 'pd_cond_probs.nsp.p'), 'wb') as f:
                pickle.dump(pd_cond_probs, f)

        except Exception as e:
            print(f"{e}")
        


        


if __name__ == '__main__':

    embeder = PackedBert()
    embeder.code2embeds()