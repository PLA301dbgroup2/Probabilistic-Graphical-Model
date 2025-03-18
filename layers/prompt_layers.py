import torch
import torch.nn as nn
from layers.embedder import PackedBert


class CodePrompt(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.prompt_num = args.prompt_num
        self.prompt_embeddings = nn.Embedding(self.prompt_num, args.hidden_size)

    def forward(self, batch_size):
        """
        used for addingg to the end of the original inputs
        """
        device = next(self.prompt_embeddings.parameters()).device
        prompt_embeddings = self.prompt_embeddings(torch.tensor([i for i in range(self.prompt_num)]).to(device)).repeat([batch_size, 1, 1])
        prompt_mask = torch.ones(batch_size, self.prompt_num).to(device)
        return prompt_embeddings, prompt_mask

class EdgePrompt(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, batch_size):
        pass
        

class LabelPrompt(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.expired_label_embedding = nn.Embedding(2, args.hidden_size)
        self.readmittion_label_embedding = nn.Embedding(2, args.hidden_size)

        self.los_label_embedding = nn.Embedding(5, args.hidden_size)


       
        self.trans = nn.Linear(768, args.hidden_size)
        if args.init_label_prompt_by_bert:
            bert = PackedBert()
            expired_embedding = bert.get_seq_embedding('expired') # 1 * 768
            un_expired_embedding = bert.get_seq_embedding('unexpired')
            readmittion_embedding = bert.get_seq_embedding('readmittion') # 1 * 768
            discharge_embedding = bert.get_seq_embedding('discharge') # 1 * 768
            los_label_embeddings = []

            los_prompts = [
                'patient\'s length of hospital stay is approximately zero to three days',
                'patient\'s length of hospital stay is approximately four to seven days',
                'patient\'s length of hospital stay is approximately eight to fourteen days',
                'patient\'s length of hospital stay is approximately fifteen to thirty days',
                'patient\'s length of hospital stay is approximately more than thirty days'
            ]
            for prompt in los_prompts: 
                los_label_embeddings.append(bert.get_seq_embedding(prompt))

            del bert
            if args.dim_reduction == 'truncate':
                self.expired_label_embedding.weight = nn.Parameter(torch.stack([expired_embedding[:, 0: args.hidden_size], un_expired_embedding[:, 0: args.hidden_size]]).squeeze())
                self.readmittion_label_embedding.weight = nn.Parameter(torch.stack([readmittion_embedding[:, 0: args.hidden_size], discharge_embedding[:, 0: args.hidden_size]]).squeeze())
                self.los_label_embedding.weight = nn.Parameter(torch.stack(los_label_embeddings).squeeze()[:, 0: args.hidden_size])
            elif args.dim_reduction == 'linaer_trans':
                self.expired_label_embedding.weight = self.trans(torch.stack([expired_embedding, un_expired_embedding]))
                self.readmittion_label_embedding.weight = self.trans(torch.stack([readmittion_embedding, discharge_embedding]))
                self.los_label_embedding.weight = self.trans(torch.stack(los_label_embeddings))
            else: assert False, f'dim_reduction {args.dim_reduction} is unrecoginzied, it must be \'truncate\' or \'linaer_trans\'' 

    def forward(self, batch_size, label_key):
        """
        used for addingg to the end of the original inputs, and need to compute dot production of vsist and label
        """
        device = next(self.expired_label_embedding.parameters()).device
        if label_key == 'expired':
           label_prompt =  self.expired_label_embedding(torch.tensor([0, 1]).to(device)).repeat(batch_size, 1, 1)
           label_prompt_mask = torch.ones(batch_size, 2).to(device)
           return label_prompt, label_prompt_mask
        elif label_key == 'readmittion':
            label_prompt =  self.expired_label_embedding(torch.tensor([0, 1]).to(device)).repeat(batch_size, 1, 1)
            label_prompt_mask = torch.ones(batch_size, 2).to(device)
            return label_prompt, label_prompt_mask
        elif label_key == 'los':

            label_prompt =  self.los_label_embedding(torch.tensor([range(5)]).to(device)).repeat(batch_size, 1, 1)
            label_prompt_mask = torch.ones(batch_size, 5).to(device)
            return label_prompt, label_prompt_mask
        else:
            assert False, f'label_key {label_key} is unrecoginzied'