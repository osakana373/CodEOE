from transformers import AutoModel, AutoConfig
# from src.common import MultiHeadAttention

import torch
import torch.nn as nn
from itertools import accumulate
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)
    
        self.dense_layers = nn.ModuleDict({
            'ent': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 4),
            'rel': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 3),
            'pol': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 4)
        })
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.cfg = cfg
        self.coAttention = ConvAttention(hid_dim=768, n_heads=1, pre_channels=None, channels=128, groups=1, layers=3, dropout=0.1)

        self.hid2hid = nn.Linear(768 + 128, 768)
        self.elu = nn.ELU()
        self.ln = nn.LayerNorm(768)

    def merge_sentence(self, sequence_outputs, input_masks, dialogue_length):
        res = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                stack.append(sequence_outputs[j, :lens])
            res.append(torch.cat(stack))
        new_res = sequence_outputs.new_zeros([len(res), max(map(len, res)), sequence_outputs.shape[-1]])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res
    
    def merge_masks(self, input_masks, dialogue_length):
        res = []
        lengths = []
        ends = list(accumulate(dialogue_length))
        starts = [w - z for w, z in zip(ends, dialogue_length)]
        for i, (s, e) in enumerate(zip(starts, ends)):
            stack = []
            length = []
            for j in range(s, e):
                lens = input_masks[j].sum()
                length.append(lens.item())
                stack.append(input_masks[j][:lens])
            res.append(torch.cat(stack))
            lengths.append(length)
        new_res = input_masks.new_zeros([len(res), max(map(len, res))])
        for i, w in enumerate(res):
            new_res[i, :len(w)] = w
        return new_res, lengths
    
    def custom_sinusoidal_position_embedding(self, token_index, pos_type):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.cfg.inner_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.cfg.device)
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings
    
    def get_instance_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index, thread_length, pos_type):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """

        seq_len, num_classes = qw.shape[:2]

        accu_index = [0] + list(accumulate(thread_length))

        logits = qw.new_zeros([seq_len, seq_len, num_classes])

        for i in range(len(thread_length)):
            for j in range(len(thread_length)):
                rstart, rend = accu_index[i], accu_index[i+1]
                cstart, cend = accu_index[j], accu_index[j+1]

                cur_qw, cur_kw = qw[rstart:rend], kw[cstart:cend]

                x, y = token_index[rstart:rend], token_index[cstart:cend]
                # This is used to compute relative distance, see the matrix in Fig.8 of our paper
                x = - x if i > 0 and i < j else x
                y = - y if j > 0 and i > j else y

                x_pos_emb = self.custom_sinusoidal_position_embedding(x, pos_type)
                y_pos_emb = self.custom_sinusoidal_position_embedding(y, pos_type)

                # Refer to https://kexue.fm/archives/8265
                x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                x_sin_pos = x_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
                cur_qw2 = torch.stack([-cur_qw[..., 1::2], cur_qw[..., ::2]], -1)
                cur_qw2 = cur_qw2.reshape(cur_qw.shape)
                cur_qw = cur_qw * x_cos_pos + cur_qw2 * x_sin_pos

                y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
                y_sin_pos = y_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
                cur_kw2 = torch.stack([-cur_kw[..., 1::2], cur_kw[..., ::2]], -1)
                cur_kw2 = cur_kw2.reshape(cur_kw.shape)
                cur_kw = cur_kw * y_cos_pos + cur_kw2 * y_sin_pos

                pred_logits = torch.einsum('mhd,nhd->mnh', cur_qw, cur_kw).contiguous()
                logits[rstart:rend, cstart:cend] = pred_logits

        return logits 

    def get_ro_embedding(self, qw, kw, token_index, thread_lengths, pos_type):
        # qw_res = qw.new_zeros(*qw.shape)
        # kw_res = kw.new_zeros(*kw.shape)
        logits = []
        batch_size = qw.shape[0]
        for i in range(batch_size):
            pred_logits = self.get_instance_embedding(qw[i], kw[i], token_index[i], thread_lengths[i], pos_type)
            logits.append(pred_logits)
        logits = torch.stack(logits) 
        return logits 
    
    def classify_matrix(self, kwargs, sequence_outputs, mat_name='ent'):

        utterance_index, token_index, thread_lengths = [kwargs[w] for w in ['utterance_index', 'token_index', 'thread_lengths']]
        input_labels = kwargs[f"{mat_name}_matrix"]
        # print(input_labels)
        masks = kwargs['sentence_masks'] if mat_name == 'ent' else kwargs['full_masks']

        dense_layer = self.dense_layers[mat_name]

        outputs = dense_layer(sequence_outputs)
        outputs = torch.split(outputs, self.cfg.inner_dim * 4, dim=-1)
        outputs = torch.stack(outputs, dim=-2)

        q_token, q_utterance, k_token, k_utterance = torch.split(outputs, self.cfg.inner_dim, dim=-1)
        #RoPE
        pred_logits = self.get_ro_embedding(q_token, k_token, token_index, thread_lengths, pos_type=0) # pos_type=0 for token-level relative distance encoding
        if mat_name != 'ent':
            pred_logits1 = self.get_ro_embedding(q_utterance, k_utterance, utterance_index, thread_lengths, pos_type=1) # pos_type=1 for utterance-level relative distance encoding
            pred_logits += pred_logits1

        nums = pred_logits.shape[-1]

        criterion = nn.CrossEntropyLoss(sequence_outputs.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)

        return loss, pred_logits 

    def forward(self, **kwargs):
        input_ids, input_masks, input_segments = kwargs['input_ids'], kwargs['input_masks'], kwargs['input_segments']
        document_length = kwargs['document_length']

        sequence_output = self.bert(input_ids, attention_mask=input_masks, token_type_ids=input_segments)[0]
        if self.training:
            sequence_output = self.dropout(sequence_output)

        # grid-tagging
        sequence_output = self.merge_sentence(sequence_output, input_masks, document_length)
        loss0, tags0 = self.classify_matrix(kwargs, sequence_output, 'ent')
        loss1, tags1 = self.classify_matrix(kwargs, sequence_output, 'rel')
        loss2, tags2 = self.classify_matrix(kwargs, sequence_output, 'pol')

        return (loss0, loss1, loss2), (tags0, tags1, tags2)

