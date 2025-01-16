import numpy as np
from collections import defaultdict

import os
import random
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def update_config(config):
    lang = config.lang
    keys = ['json_path']
    for k in keys:
        config[k] = config[k] + '_' + lang
    keys = ['cls', 'sep', 'pad', 'unk', 'bert_path']
    for k in keys:
        config[k] = config['bert-' + config.lang][k]
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

class WordPair:
    def __init__(self, max_sequence_len=512):
        self.max_sequence_len = max_sequence_len
        self.entity_dic = {"O": 0, "ENT-Tri": 1, "ENT-Arg": 2, "ENT-Opi": 3}
        self.rel_dic = {"O": 0, "h2h": 1, "t2t": 2}
        self.polarity_dic = {"O": 0, "positive": 1, "negative": 2, "neutral": 3}
    
    def encode_entity(self, elements, entity_type='ENT-Tri'):
        """
        Convert the elements in the dataLoader to a list of entities rel_list.
        The format is [(start, end, entity_type in the dictionary), ...]
        """
        entity_list = []
        for line in elements:
            start, end = line[:2]
            entity_list.append((start, end, self.entity_dic[entity_type]))

        return entity_list

    def encode_relation(self, quadruples):
        """
        Convert the quadruples in the dataLoader to a list of relations rel_list.
        The format is [(start, end, relation_type in the dictionary), ...]
        """
        rel_list = []
        # quad: [trigger_start, trigger_end, arg_list, opinion_start, opinion_end, sentiment]
        for quad in quadruples:
            trigger_start, trigger_end = quad[:2]
            arg_list = quad[2]
            opinion_start, opinion_end = quad[3:5]
            # add trigger-arg relations, arg_list is a list of [arg_start, arg_end]
            if trigger_start != -1:
                for arg in arg_list:
                    arg_start, arg_end = arg
                    if arg_start != -1:   
                        rel_list.append((trigger_start, arg_start, self.rel_dic['h2h']))
                        rel_list.append((trigger_end, arg_end, self.rel_dic['t2t']))
            if trigger_start != - 1 and opinion_start != -1:
                rel_list.append((trigger_start, opinion_start, self.rel_dic['h2h']))
                rel_list.append((trigger_end, opinion_end, self.rel_dic['t2t']))

        return rel_list

    def encode_sentiment(self, quadruples):
        """
        Convert the quadruples in the dataLoader to a list of polarity rel_list.
        The format is [(start, end, polarity in the dictionary), ...]
        """
        sentiment_list = []

        for quad in quadruples:
            trigger_start, trigger_end, arg_list, opinion_start, opinion_end, sentiment = quad
            sentiment_list.append((trigger_start, opinion_start, sentiment))
            sentiment_list.append((trigger_end, opinion_end, sentiment))
        
        return sentiment_list

    def list2rel_matrix4batch(self, batch_rel_list, seq_len=512):
        """
        Convert the list of relations to a relation matrix.
        batch_rel_matrix:[batch_size, seq_len, seq_len]
        """
        rel_matrix = np.zeros([len(batch_rel_list), seq_len, seq_len], dtype=int)
        for batch_id, rel_list in enumerate(batch_rel_list):
            for rel in rel_list:
                rel_matrix[batch_id, rel[0], rel[1]] = rel[2]
        return rel_matrix.tolist()
    
    def rel_matrix2list(self, rel_matrix):
        """
        Convert a (512*512) matrix to a list of relations.
        """
        rel_list = []
        nonzero = rel_matrix.nonzero()
        for x_index, y_index in zip(*nonzero):
            dic_key = int(rel_matrix[x_index][y_index].item())
            rel_elem = (x_index, y_index, dic_key)
            rel_list.append(rel_elem)
        return rel_list
    
    def get_quadruples(self, ent_matrix, rel_matrix, pol_matrix, new_sentences_ids):
        ent_list = self.rel_matrix2list(ent_matrix)
        rel_list = self.rel_matrix2list(rel_matrix)
        pol_list = self.rel_matrix2list(pol_matrix)
        res, pair = self.decode_quadruple(ent_list, rel_list, pol_list, new_sentences_ids)
        return res, pair
    
    def decode_quadruple(self, ent_list, rel_list, pol_list, new_sentences_ids):
        """
        Decode the entity, relation, polarity list to a list of quadruples.
        """
        # decode entity
        entity_elem_dic = defaultdict(list)
        entity2type = {}
        for entity in ent_list:
            # avoid extract span from two different documents
            if new_sentences_ids[entity[0]] != new_sentences_ids[entity[1]]:
                continue
            entity_elem_dic[entity[0]].append((entity[1], entity[2]))
            entity2type[entity[:2]] = entity[2]
        # (boundary, boundary -> polarity) set
        b2b_relation_set = {}
        for rel in pol_list:
            b2b_relation_set[rel[:2]] = rel[-1]
        
        # tail2tail set
        t2t_relation_set = set()
        for rel in rel_list:
            if rel[2] == self.rel_dic['t2t']:
                t2t_relation_set.add(rel[:2])

        # head2head dictionary, with structure (head1: [(head2, relation type)])
        h2h_entity_elem = defaultdict(list)
        for h2h_rel in rel_list:
            # for each head-to-head relationship, mark its entity as 0
            if h2h_rel[2] != self.rel_dic['h2h']: continue
            h2h_entity_elem[h2h_rel[0]].append((h2h_rel[1], h2h_rel[2]))
        
        # for all head-to-head relations
        quadruples = []
        for h1, values in h2h_entity_elem.items():
            if h1 not in entity_elem_dic: continue
            for h2, rel_tp in values:
                if h2 not in entity_elem_dic: continue
                for t1, ent1_tp in entity_elem_dic[h1]:
                    for t2, ent2_tp in entity_elem_dic[h2]:
                        if (t1, t2) in t2t_relation_set:
                            quadruples.append((h1, t1, h2, t2))
        if (0,0,0,0) in quadruples:
            quadruples.remove((0,0,0,0))
        quadruple_set = set(quadruples)
        ele2list = defaultdict(list)
        for line in quadruples:
            e0, e1 = line[:2], line[2:]
            ele2list[e0].append(e1)  
        
        pairs = {'trigger-arg': [], 'trigger-opinion': []}
        for line in quadruple_set:
            h1, t1, h2, t2 = line
            tp1 = entity2type[(h1, t1)]
            tp2 = entity2type[(h2, t2)]
            if tp1 == 1 and tp2 == 2:
                pairs['trigger-arg'].append(line)
            elif tp1 == 1 and tp2 == 3:
                pairs['trigger-opinion'].append(line)
            elif tp2 == 1 and tp1 == 2:
                pairs['trigger-arg'].append((h2, t2, h1, t1))
            elif tp2 == 1 and tp1 == 3:
                pairs['trigger-opinion'].append((h2, t2, h1, t1))

        tri_opi_senti_pairs = []
        for ele1_s, ele1_e, ele2_s, ele2_e in pairs['trigger-opinion']:
            senti1 = b2b_relation_set.get((ele1_s, ele2_s), -1)
            senti2 = b2b_relation_set.get((ele1_e, ele2_e), -1)
            if (senti1 == senti2 or senti1 == -1) and senti2 != -1:
                tri_opi_senti_pairs.append((ele1_s, ele1_e, ele2_s, ele2_e, senti2))
            elif senti1 != -1 and senti2 == -1:
                tri_opi_senti_pairs.append((ele1_s, ele1_e, ele2_s, ele2_e, senti1))
        return tri_opi_senti_pairs, pairs

class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []
        
        
