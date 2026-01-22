#!/usr/bin/env python

import torch
import numpy as np
from attrdict import AttrDict
from scipy.linalg import block_diag
from collections import defaultdict
from attrdict import AttrDict 

from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import random
from loguru import logger
import json

from src.common import WordPair
from src.preprocess import Preprocessor
from src.run_eval import Run_eval

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataLoader:
    def __init__(self, cfg):
        path = os.path.join(cfg.preprocessed_dir, '{}_{}.pkl'.format(cfg.lang, cfg.bert_path.replace('/', '-')))
        preprocessor = Preprocessor(cfg)

        data = None
        if not os.path.exists(path):
            logger.info('Preprocessing data...')
            data = preprocessor.forward()
            logger.info('Saving preprocessed data to {}'.format(path))
            if not os.path.exists(cfg.preprocessed_dir):
                os.makedirs(cfg.preprocessed_dir)
            pkl.dump(data, open(path, 'wb'))
        
        logger.info('Loading preprocessed data from {}'.format(path))
        self.data = pkl.load(open(path, 'rb')) if data is None else data

        self.kernel = WordPair()
        self.config = cfg
    
    def worker_init(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def getdata(self):
        load_data = lambda mode: DataLoader(MyDataset(self.data[mode]), num_workers=0, worker_init_fn=self.worker_init,
                                            shuffle=(mode == 'train'), batch_size=self.config.batch_size, 
                                            collate_fn=self.collate_fn)
        train_loader, valid_loader, test_loader = map(load_data, ['train', 'valid', 'test']) 
        res = (train_loader, valid_loader, test_loader, self.config)

        return res

    def collate_fn(self, lst):
        doc_id, input_ids, input_masks, input_segments, sentence_length, \
                        new_sentences_ids, utterance_index, token_index, thread_length, \
                        pieces2words, new2old, new_quadruples, \
                        pairs, entity_list, relation_list, polarity_list = zip(*lst)
        document_length = list(map(len, input_ids))
        
        # padding
        max_lens = max(map(lambda line: max(map(len, line)), input_ids))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])

        # padding for new_sentences_ids
        max_lens = max(map(len, new_sentences_ids))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for w in input_batch]
        new_sentences_ids, utterance_index, token_index = map(padding, [new_sentences_ids, utterance_index, token_index])

        # padding for entity, relation, polarity lists
        padding = lambda input_batch: [list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for w in input_batch]
        entity_lists, relation_lists, polarity_lists = map(padding, [entity_list, relation_list, polarity_list])

        # padding for quadruples
        max_tri_num = max(map(len, new_quadruples))
        quadruple_masks = [[1] * len(w) + [0] * (max_tri_num - len(w)) for w in new_quadruples]
        quadruples = [list(map(list, w)) + [[0] * 6] * (max_tri_num - len(w)) for w in new_quadruples]

        # padding for sentences
        sentence_masks = np.zeros([len(new_sentences_ids), max_lens, max_lens], dtype=int)
        for i in range(len(sentence_length)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length[i]]
            masks = block_diag(*masks)
            sentence_masks[i, :len(masks), :len(masks)] = masks
        sentence_masks = sentence_masks.tolist()

        flatten_length = list(map(sum, sentence_length))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(np.int64)
        full_masks = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()

        entity_matrix = self.kernel.list2rel_matrix4batch(entity_lists, max_lens)
        relation_matrix = self.kernel.list2rel_matrix4batch(relation_lists, max_lens)
        polarity_matrix = self.kernel.list2rel_matrix4batch(polarity_lists, max_lens)
        
        res = {
            'doc_id': doc_id,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'input_segments': input_segments,
            'ent_matrix': entity_matrix,
            'rel_matrix': relation_matrix,
            'pol_matrix': polarity_matrix,
            'sentence_masks': sentence_masks,
            'full_masks': full_masks,
            'quadruples': quadruples,
            'quadruple_masks': quadruple_masks,
            'pairs': pairs,
            'new_sentences_ids': new_sentences_ids,
            'document_length': document_length,
            'utterance_index': utterance_index,
            'token_index': token_index,
            'thread_lengths': thread_length,
            'pieces2words': pieces2words,
            'new2old': new2old
        }

        nocuda = ['quadruples', 'pairs', 'doc_id', 'pieces2words', 'new2old', 'thread_lengths']
        res = {k : v if k in nocuda else torch.tensor(v).to(self.config.device) for k, v in res.items()}

        return res

class RelationMetric:
    def __init__(self, config):
        self.clear()
        self.kernel = WordPair()
        self.predict_result = defaultdict(list)
        self.config = config
    
    def clear(self):
        self.predict_result = defaultdict(list)

    def trans2position(self, triplet, new2old, pieces2words):
        res = []
        """
        recover the position of entities in the original sentence

        new2old: transfer position from index with CLS and SEP to index without CLS and SEP
        pieces2words: transfer position from index of wordpiece to index of original words 

        Example:
        list0 (original sentence):"London is the capital of England"
        list1 (tokenized sentence): "Lon ##don is the capital of England"
        list2 (packed sentence): "[CLS] Lon #don is the capital of England [SEP]"
        predicted entity: (1, 2), denotes "Lon #don" in list2

        new2old: list2->list1
          = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, ...}
        pieces2words: list1->list0
          = {'0': 0, '1': 0, '2': 1, '3': 2, '4': 3, ...}

        input  -> entity in list2: "Lon #don" (1, 2)
        middle -> entity in list1: "Lon #don" (0, 1)
        output -> entity in list0: "London"   (0, 0)
        """

        head = lambda x : pieces2words[new2old[x]]
        tail = lambda x : pieces2words[new2old[x]]

        triplet = list(triplet)
        for s0, e0, s1, e1, s2, e2, pol in triplet:
            ns0, ns1, ns2 = head(s0), head(s1), head(s2)
            ne0, ne1, ne2 = tail(e0), tail(e1), tail(e2)
            res.append([ns0, ne0, ns1, ne1, ns2, ne2, pol])
        return res
    
    def trans2pair(self, pred_pairs, new2old, pieces2words):
        new_pairs = {}
        new_pos = lambda x : pieces2words[new2old[x]]
        for k, line in pred_pairs.items():
            new_line = []
            for s0, e0, s1, e1 in line:
                s0, e0, s1, e1 = new_pos(s0), new_pos(e0), new_pos(s1), new_pos(e1)
                new_line.append([s0, e0, s1, e1])
            new_pairs[k] = new_line
        return new_pairs
    
    def opinion_trans2pair(self, pred_opinion_senti_pairs, new2old, pieces2words):
        new_pairs = {}
        new_pos = lambda x : pieces2words[new2old[x]]
        for line in pred_opinion_senti_pairs:
            new_line = []
            for start, end, sentiment in line:
                start, end = new_pos(start), new_pos(end)
                new_line.append([start, end, sentiment])
        return new_pairs

    def filter_entity(self, ent_list, new2old, pieces2words):
        res = []
        for s, e, pol in ent_list:
            ns, ne = pieces2words[new2old[s]], pieces2words[new2old[e]]
            res.append([ns, ne, pol])
        return res
    
    def tri_opi_senti_trans2pair(self, pred_tri_opi_senti_pairs, new2old, pieces2words):
        new_pairs = []
        new_pos = lambda x : pieces2words[new2old[x]]
        for s1, e1, s2, e2, senti in pred_tri_opi_senti_pairs:
            s1, e1, s2, e2 = new_pos(s1), new_pos(e1), new_pos(s2), new_pos(e2)
            new_pairs.append([s1, e1, s2, e2, senti])
        return new_pairs 

    def add_instance(self, data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix):
        """
        input_matrix: [batch_size, seq_len, seq_len]
        pred_matrix: [batch_size, seq_len, seq_len, 4]
        input_masks: [batch_size, seq_len]
        """

        pred_ent_matrix = pred_ent_matrix.argmax(-1) * data['sentence_masks']
        pred_rel_matrix = pred_rel_matrix.argmax(-1) * data['full_masks']
        pred_pol_matrix = pred_pol_matrix.argmax(-1) * data['full_masks'] 
        new_sentences_ids = data['new_sentences_ids'].tolist()
        new2old = data['new2old']
        pieces2words = data['pieces2words']
        doc_id = data['doc_id']

        pred_rel_matrix = np.array(pred_rel_matrix.tolist())
        pred_ent_matrix = np.array(pred_ent_matrix.tolist())
        pred_pol_matrix = np.array(pred_pol_matrix.tolist())

        for i in range(len(pred_ent_matrix)):
            ent_matrix, rel_matrix, pol_matrix = pred_ent_matrix[i], pred_rel_matrix[i], pred_pol_matrix[i]
            
            pred_tri_opi_senti_pairs, pred_pairs = self.kernel.get_quadruples(ent_matrix, rel_matrix, pol_matrix, new_sentences_ids[i])
            
            pred_ents = self.kernel.rel_matrix2list(ent_matrix)

            pred_ents = self.filter_entity(pred_ents, new2old[i], pieces2words[i])
            pred_pairs = self.trans2pair(pred_pairs, new2old[i], pieces2words[i]) 
            pred_tri_opi_senti_pairs = self.tri_opi_senti_trans2pair(pred_tri_opi_senti_pairs, new2old[i], pieces2words[i])

            self.predict_result[doc_id[i]].append(pred_ents)
            self.predict_result[doc_id[i]].append(pred_pairs)
            self.predict_result[doc_id[i]].append(pred_tri_opi_senti_pairs)
    
    def save2file(self, gold_file, pred_file, lang):
        pol_dict = self.config.polarity_dict
        reverse_pol_dict = {v: k for k, v in pol_dict.items()}
        reverse_ent_dict = {v: k for k, v in self.config.entity_dict.items()}

        gold_file = open(gold_file, 'r', encoding='utf-8')
        data = json.load(gold_file)

        res = []

        for line in data:
            doc_id, sentences = line['doc_id'], line['sentences']
            if doc_id not in self.predict_result:
                continue
            doc = line['text']
            

            prediction = self.predict_result[doc_id]
            entities = defaultdict(list)
            for head, tail, tp in prediction[0]:
                tp = reverse_ent_dict[tp]
                head, tail = head, tail + 1
                tp_dict = {'ENT-Tri': 'Trigger', 'ENT-Arg': 'Argument', 'ENT-Opi': 'Opinion'}
                if lang == 'zh':
                    entities[tp_dict[tp]].append([head, tail, ''.join(doc[head:tail])])
                elif lang == 'en':
                    entities[tp_dict[tp]].append([head, tail, ' '.join(doc[head:tail])])
                else:
                    assert False, 'Invalid language'
                    
            
            pairs = defaultdict(list)
            for key in ['trigger-arg', 'trigger-opinion']:
                for s0, e0, s1, e1 in prediction[1][key]:
                    e0, e1 = e0 + 1, e1 + 1
                    if lang == 'zh':
                        pairs[key].append([s0, e0, s1, e1, ''.join(doc[s0:e0]), ''.join(doc[s1:e1])])
                    elif lang == 'en':
                        pairs[key].append([s0, e0, s1, e1, ' '.join(doc[s0:e0]), ' '.join(doc[s1:e1])])
                    else:
                        assert False, 'Invalid language'

            
            tri_opi_senti_pairs = []
            for s0, e0, s1, e1, senti in prediction[2]:
                e0, e1 = e0 + 1, e1 + 1
                if lang == 'zh':
                    tri_opi_senti_pairs.append([s0, e0, s1, e1, reverse_pol_dict[senti], ''.join(doc[s0:e0]), ''.join(doc[s1:e1])])
                elif lang == 'en':
                    tri_opi_senti_pairs.append([s0, e0, s1, e1, reverse_pol_dict[senti], ' '.join(doc[s0:e0]), ' '.join(doc[s1:e1])])
                else:
                    assert False, 'Invalid language'

            full_quadruples = []
            new_quadruples = []
            for tri_s, tri_e, opi_s, opi_e, tri_text, opi_text in pairs['trigger-opinion']:
                sentiment = ''
                for t_s, t_e, o_s, o_e, senti, t_text, o_text in tri_opi_senti_pairs:
                    if tri_text == t_text and opi_text == o_text:
                        sentiment = senti
                        break
                # find all arguments of the trigger and opinion in pairs['trigger-arg'] and pairs['arg-opinion']
                arg_list = []
                for s0, e0, s1, e1, s0_text, s1_text in pairs['trigger-arg']:
                    if s0_text == tri_text:
                        if lang == 'zh':
                            arg_list.append([s1, e1, s1_text])
                        elif lang == 'en':
                            arg_list.append([s1, e1, s1_text])
                # remove duplicate arguments
                arg_list = list(set([tuple(w) for w in arg_list]))

                new_quadruples.append([tri_s, tri_e, tri_text, arg_list, opi_s, opi_e, opi_text, sentiment])
                full_quadruples.append([tri_s, tri_e, tri_text, arg_list, opi_s, opi_e, opi_text, sentiment])

            res.append({
                'doc_id': doc_id,
                'quadruples': new_quadruples,
                'triggers': entities['Trigger'],
                'arguments': entities['Argument'],
                'opinions': entities['Opinion'],
                'trigger-arg': pairs['trigger-arg'],
                'trigger-opinion': pairs['trigger-opinion'],
                # 'arg-opinion': pairs['arg-opinion'],
                'full_quadruples': full_quadruples,
                'tri_opi_senti_pairs': tri_opi_senti_pairs
                })
        logger.info('Save prediction results to {}'.format(pred_file))
        json.dump(res, open(pred_file, 'w', encoding='utf-8'), ensure_ascii=False)

    def compute(self, name='valid', epoch=0):
        # action: pred, make prediction, save to file 
        # action: eval, make prediction, save to file and evaluate 

        args = AttrDict({
            'pred_file': os.path.join(self.config.target_dir, 'pred_{}_{}_epoch{}_layer_4.json'.format(self.config.lang, name, epoch)),
            'gold_file': os.path.join(self.config.json_path, '{}.json'.format(name))
        })
        self.save2file(args.gold_file, args.pred_file, self.config.lang)

        tri_arg_score, tri_opi_senti_score, res = Run_eval(args).forward(lang=self.config.lang)
        self.clear()
        return tri_arg_score[2], tri_opi_senti_score[2], res