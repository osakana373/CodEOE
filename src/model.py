from transformers import AutoModel, AutoConfig
# from src.common import MultiHeadAttention

import torch
import torch.nn as nn
from itertools import accumulate
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class ConvAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, groups, dropout=0.1):
        super(ConvAttentionLayer, self).__init__()
        assert hid_dim % n_heads == 0
        self.n_heads = n_heads
        if pre_channels is not None:
            input_channels = hid_dim * 2 + pre_channels 
        else:
            input_channels = hid_dim * 2 
        # input_channels = hid_dim * 2
        self.groups = groups

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.linear1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=False)

        # self.seq_batch_size = 32
        # input_channels = hid_dim * 2 / self.seq_batch_size

        self.conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_channels, channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.score_layer = nn.Conv2d(channels, n_heads, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.window_size = 256


    def forward(self, x, y, pre_conv=None, mask=None, residual=True, self_loop=True):
        ori_x, ori_y = x, y

        B, M, _ = x.size()
        B, N, _ = y.size()
        
        # # 滑动窗口
        # out_x = torch.zeros_like(x)
        # out_y = torch.zeros_like(y)

        # for i in range(0, M, self.window_size // 2):
        #     win_x_start = max(0, i - self.window_size // 2)
        #     win_x_end = min(M, i + self.window_size // 2)
        #     win_y_start = max(0, i - self.window_size // 2)
        #     win_y_end = min(N, i + self.window_size // 2)

        #     win_x = x[:, win_x_start:win_x_end, :]
        #     win_y = y[:, win_y_start:win_y_end, :]

        #     fea_map = torch.cat([win_x.unsqueeze(2).repeat_interleave(win_y.size(1), 2), win_y.unsqueeze(1).repeat_interleave(win_x.size(1), 1)],
        #                         -1).permute(0, 3, 1, 2).contiguous()
        #     # if pre_conv is not None:
        #     #     fea_map = torch.cat([fea_map, pre_conv[:, :, win_x_start:win_x_end, win_y_start:win_y_end]], 1)
        #     fea_map = self.conv(fea_map)
        #     scores = self.activation(self.score_layer(fea_map))

        #     if mask is not None:
        #         # mask = mask.expand_as(scores)
        #         # scores = scores.masked_fill(mask.eq(0), -6e4)
        #         # mask也要切片处理
        #         win_mask = mask[:, :, win_x_start:win_x_end, win_y_start:win_y_end]
        #         win_mask = win_mask.expand_as(scores)
        #         scores = scores.masked_fill(win_mask.eq(0), -6e4)



        #     win_x = self.linear1(self.dropout(win_x))
        #     win_y = self.linear2(self.dropout(win_y))
        #     win_x_out = torch.matmul(F.softmax(scores, -1), win_y.view(B, win_x.size(1), self.n_heads, -1).transpose(1, 2))
        #     win_x_out = win_x_out.transpose(1, 2).contiguous().view(B, win_x.size(1), -1)
        #     win_y_out = torch.matmul(F.softmax(scores.transpose(2, 3), -1), win_x.view(B, win_y.size(1), self.n_heads, -1).transpose(1, 2))
        #     win_y_out = win_y_out.transpose(1, 2).contiguous().view(B, win_y.size(1), -1)

        #     if self_loop:
        #         win_x_out = win_x_out + win_x
        #         win_y_out = win_y_out + win_y
            
        #     win_x_out = self.activation(win_x_out)
        #     win_y_out = self.activation(win_y_out)

        #     if residual:
        #         win_x_out = win_x_out + ori_x[:, win_x_start:win_x_end, :]
        #         win_y_out = win_y_out + ori_y[:, win_y_start:win_y_end, :]
            
        #     out_x[:, win_x_start:win_x_end, :] = win_x_out
        #     out_y[:, win_y_start:win_y_end, :] = win_y_out
        #     # if global_fea_map is None:
        #     #     global_fea_map = fea_map
        #     # else:
        #     #     # 最后两维拼接
        #     #     global_fea_map = torch.cat([global_fea_map, fea_map], -1)
        # return out_x, out_y, fea_map

        # M, _ = x.size()
        # N, _ = y.size()
        # fea_map = torch.cat([x.unsqueeze(1).repeat_interleave(N, 1), y.unsqueeze(0).repeat_interleave(M, 0)], -1).permute(2, 0, 1).contiguous()
        fea_map = torch.cat([x.unsqueeze(2).repeat_interleave(N, 2), y.unsqueeze(1).repeat_interleave(M, 1)],
                            -1).permute(0, 3, 1, 2).contiguous()
        if pre_conv is not None:
            fea_map = torch.cat([fea_map, pre_conv], 1)
            # fea_map = torch.cat([fea_map, pre_conv], 0)
        fea_map = self.conv(fea_map)
        scores = self.activation(self.score_layer(fea_map))
        
        if mask is not None:
            mask = mask.expand_as(scores)
            scores = scores.masked_fill(mask.eq(0), -9e10)

        x = self.linear1(self.dropout(x))
        y = self.linear2(self.dropout(y))
        
        out_x = torch.matmul(F.softmax(scores, -1), y.view(B, N, self.n_heads, -1).transpose(1, 2))
        out_x = out_x.transpose(1, 2).contiguous().view(B, M, -1)
        out_y = torch.matmul(F.softmax(scores.transpose(2, 3), -1), x.view(B, M, self.n_heads, -1).transpose(1, 2))
        out_y = out_y.transpose(1, 2).contiguous().view(B, N, -1)

        out_x = self.activation(out_x)
        out_y = self.activation(out_y)

        if residual:
            out_x = out_x + ori_x
            out_y = out_y + ori_y
        return out_x, out_y, fea_map


class ConvAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, pre_channels, channels, layers, groups, dropout):
        super(ConvAttention, self).__init__()
        # self.layers = nn.ModuleList([ConvAttentionLayer(hid_dim , n_heads, pre_channels if i == 0 else channels,
        #                                                 channels, groups, dropout=dropout) for i in range(layers)])
        self.layers = nn.ModuleList([ConvAttentionLayer(512, n_heads, pre_channels if i == 0 else channels,
                                                        channels, groups, dropout=dropout) for i in range(layers)])
        # self.linear1 = nn.Linear(768, 512)
        # self.linear2 = nn.Linear(512, 768)
    def forward(self, x, y, fea_map=None, mask=None, residual=True, self_loop=True):
        # x = self.linear1(x)
        # y = self.linear1(y)
        for layer in self.layers:
            x, y, fea_map = layer(x, y, fea_map, mask, residual, self_loop)

        # x = self.linear2(x)
        # y = self.linear2(y)
        return x, y, fea_map.permute(0, 2, 3, 1).contiguous()


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
        # if self.training:
        #     sequence_output = self.dropout(sequence_output)

        # co-attention部分
        # merge_masks，将每个句子的mask合并成一个，类比merge_sentence
        sequence_output = self.merge_sentence(sequence_output, input_masks, document_length)
        merge_masks, sentence_lengths = self.merge_masks(input_masks, document_length)
        
        # 将merg_masks转置
        merge_masks = merge_masks.transpose(0, 1)
        length = sequence_output.size(1)

        # sequence_output [batch_size, seq_len, hidden_size], sentence_lengths [batch_size, every sentence length]
        # 按照sentence_lengths将sequence_output分割成多个句子，依次做co-attention
        h_ner, h_re = None, None
        for i in range(len(sentence_lengths)):
            ends = list(accumulate(sentence_lengths[i]))
            starts = [w - z for w, z in zip(ends, sentence_lengths[i])]
            batch_h_ner, batch_h_re = None, None
            news_input = None
            news_end= 0
            for j, (s, e) in enumerate(zip(starts, ends)):
                # 在第0维扩展一维
                if j == 0:
                    news_input = sequence_output[i, s:e]
                    news_end = e
                    sequence_input = sequence_output[i, s:e].unsqueeze(0)
                    hb_ner, hb_re, hb_share = self.coAttention(sequence_input, sequence_input, None, None)
                    # 将x和y的第0维压缩
                    hb_ner = hb_ner.squeeze(0)
                    hb_re = hb_re.squeeze(0)
                    # 将hb_share平均池化，并加入到hb_ner和hb_re中，进行特征融合，hb_share的维度为[batch_size, seq_len, seq_len, hidden_size]，表示了句子中每个词与其他词的关系
                    hb_share = hb_share.squeeze(0)
                else:
                    # 将news_input和sequence_output[i, s:e]拼接
                    if e - s + news_end <= 512:
                        sequence_input = torch.cat((news_input, sequence_output[i, s:e]), dim=0)
                        sequence_input = sequence_input.unsqueeze(0)
                        hb_ner, hb_re, hb_share = self.coAttention(sequence_input, sequence_input, None, None)
                        # 再将除了news_input的部分提取出来
                        hb_ner = hb_ner.squeeze(0)[news_end:]
                        hb_re = hb_re.squeeze(0)[news_end:]
                        hb_share = hb_share.squeeze(0)[news_end:, news_end:, :]
                    else:
                        sequence_input = sequence_output[i, s:e].unsqueeze(0)
                        hb_ner, hb_re, hb_share = self.coAttention(sequence_input, sequence_input, None, None)
                        hb_ner = hb_ner.squeeze(0)
                        hb_re = hb_re.squeeze(0)
                        hb_share = hb_share.squeeze(0)
                    
                
                # 最大池化
                # hb_share = hb_share.mean(dim=1)
                hb_share = hb_share.max(dim=1)[0]
                hb_ner = torch.cat((hb_ner, hb_share), dim=-1)
                hb_re = torch.cat((hb_re, hb_share), dim=-1)

                hb_ner = self.ln(self.hid2hid(hb_ner))
                hb_re = self.ln(self.hid2hid(hb_re))

                hb_ner = self.elu(self.dropout(hb_ner))
                hb_re = self.elu(self.dropout(hb_re))

                if batch_h_ner is None:
                    batch_h_ner, batch_h_re = hb_ner, hb_re
                else:
                    batch_h_ner = torch.cat((batch_h_ner, hb_ner), dim=0)
                    batch_h_re = torch.cat((batch_h_re, hb_re), dim=0)
            # 将batch_h_ner, batch_h_re按照length进行padding
            batch_h_ner = F.pad(batch_h_ner, (0, 0, 0, length - batch_h_ner.size(0)))
            batch_h_re = F.pad(batch_h_re, (0, 0, 0, length - batch_h_re.size(0)))
            if h_ner is None:
                # 在index为0的位置添加一维
                h_ner, h_re = batch_h_ner.unsqueeze(0), batch_h_re.unsqueeze(0)
            else:
                # 根据merge_masks的长度，
                h_ner = torch.cat((h_ner, batch_h_ner.unsqueeze(0)), dim=0)
                h_re = torch.cat((h_re, batch_h_re.unsqueeze(0)), dim=0)
            

            
        # h_ner, h_re, _ = self.coAttention(sequence_output, sequence_output, None, padding_mask)
        del merge_masks, hb_ner, hb_re, hb_share, batch_h_ner, batch_h_re, sequence_output, sequence_input

        # if self.training:
        #     h_ner = self.dropout(h_ner)
        # if self.training:
        #     h_re = self.dropout(h_re)
        loss0, tags0 = self.classify_matrix(kwargs, h_ner, 'ent')
        del h_ner
        loss1, tags1 = self.classify_matrix(kwargs, h_re, 'rel')
        loss2, tags2 = self.classify_matrix(kwargs, h_re, 'pol')
        del h_re
        
        # grid-tagging部分
        # sequence_output = self.merge_sentence(sequence_output, input_masks, document_length)
        # loss0, tags0 = self.classify_matrix(kwargs, sequence_output, 'ent')
        # loss1, tags1 = self.classify_matrix(kwargs, sequence_output, 'rel')
        # loss2, tags2 = self.classify_matrix(kwargs, sequence_output, 'pol')

        return (loss0, loss1, loss2), (tags0, tags1, tags2)

