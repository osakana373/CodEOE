from src.utils import WordPair
import os
import re
import json

import numpy as np

from collections import defaultdict
from itertools import accumulate
from transformers import AutoTokenizer
from typing import List, Dict
from loguru import logger
from tqdm import tqdm


class Preprocessor:
    def __init__(self, config):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.wordpair = WordPair()
        self.entity_dict = self.wordpair.entity_dic


    def read_data(self, mode):
        """
        Read a JSON file, tokenize using BERT, and realign the indices of the original elements according to the tokenization results.
        """

        path = os.path.join(self.config.json_path, '{}.json'.format(mode))

        if not os.path.exists(path):
            raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

        content = json.load(open(path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_data = self.parse_data(line, mode)
            res.append(new_data)
        # 1214为验证集数据
        # delete_ids = [1544, 659, 1585, 597, 1889, 689, 1668, 1214]
        # res = [data for data in res if data['doc_id'] not in delete_ids]
        return res

    def align_index_with_list(self, sentences):
        """
        Align the index of the original elements according to the tokenization results.
        """

        pieces2word = []
        word_num = 0
        all_pieces = []
        for sentence in sentences:
            tokens = [self.tokenizer.tokenize(word) for word in sentence]
            cur_line = []
            for token in tokens:
                for piece in token:
                    pieces2word.append(word_num)
                word_num += 1
                cur_line += token
            all_pieces.append(cur_line)
            # print(tokens)
        return all_pieces, pieces2word
    
    def parse_data(self, data, mode):
        sentences = data['sentences']
        # replace unknown tokens with [UNK]
        sentences = [[self.repack_unknow(word) for word in sentence] for sentence in sentences]

        # align_index_with_list: align the index of the original elements according to the tokenization results
        new_sentences, pieces2words = self.align_index_with_list(sentences)

        word2pieces = defaultdict(list)
        for piece, word in enumerate(pieces2words):
            word2pieces[word].append(piece)
        
        data['pieces2words'] = pieces2words
        data['sentences'] = new_sentences

        # get event trigger, event arguments, opinion respectively, and align to the new index
        if mode != 'train':
            return data

        event_mentions = data['event_mentions']
        # read dic list, [{'trigger_id': , 'trigger':{}, 'arguments':[]}, ...]
        triggers = [event_mention['trigger'] for event_mention in event_mentions]
        arguments = [event_mention['arguments'] for event_mention in event_mentions]
        opinions = data['opinions']

        # convert [{'text':, 'start':, 'end':}] to ['start', 'end', 'text']
        new_triggers = []
        for trigger in triggers:
            new_triggers.append([trigger['start'], trigger['end'], trigger['text']])
        new_arguments = []
        for argument in arguments:
            if argument == []:
                continue
            new_arguments.append([[arg['start'], arg['end'], arg['text']] for arg in argument])
        # 将new_arguments展平
        new_arguments = [arg for argument in new_arguments for arg in argument]

        new_opinions = []
        for opinion in opinions:
            new_opinions.append([opinion['start'], opinion['end'], opinion['text'], opinion['sentiment']])
        
        # 去掉重复的trigger，arguments，opinions，即start，end，text都相同的
        new_triggers = self.remove_duplicate(new_triggers)
        new_arguments = self.remove_duplicate(new_arguments)
        new_opinions = self.remove_duplicate(new_opinions)

        # align the index of the original elements according to the tokenization results
        # print(pieces2words)
        # print(new_sentences)
        # print(sentences)
        new_triggers = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z) for x, y, z in new_triggers]
        new_arguments = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z) for x, y, z in new_arguments]
        # print(pieces2words)
        # print(new_opinions)
        new_opinions = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z, w) for x, y, z, w in new_opinions]

        data['triggers'], data['arguments'], data['opinions'] = new_triggers, new_arguments, new_opinions

        news = [w for line in new_sentences for w in line]


        if self.config.lang == 'zh':
            # check the index 
            for ts, te, t_t in new_triggers:
                assert self.check_text(''.join(news[ts:te]), t_t)
            for ts, te, t_t in new_arguments:
                assert self.check_text(''.join(news[ts:te]), t_t)
            for ts, te, t_t, _ in new_opinions:
                assert self.check_text(''.join(news[ts:te]), t_t)
        else:
            # check the index for english data
            for ts, te, t_t in new_triggers:
                assert self.check_text(''.join(news[ts:te]), ''.join(t_t))
            for ts, te, t_t in new_arguments:
                assert self.check_text(''.join(news[ts:te]), ''.join(t_t))
            for ts, te, t_t, _ in new_opinions:
                assert self.check_text(''.join(news[ts:te]), ''.join(t_t))
        
        # create quadruple: [tirgger, [event args1, event args2, ], opinion, sentiment]
        # data['quadruples'] = [(trigger, [argument1, argument2, ...], opinion, sentiment)]
        # according to trigger, we can find the corresponding arguments and opinions

        data['quadruples'] = []
        for element in event_mentions:
            trigger_text = element['trigger']['text']
            trigger_start = word2pieces[element['trigger']['start']][0]
            trigger_end = word2pieces[element['trigger']['end']-1][-1] + 1

            if self.config.lang == 'zh':
                assert self.check_text(''.join(news[trigger_start:trigger_end]), trigger_text)
            else:
                assert self.check_text(''.join(news[trigger_start:trigger_end]), ''.join(trigger_text))
            
            # add all arguments to argument_list, if argument is empty, add [[-1, -1, '']]
            argument_list = []
            if element['arguments'] == []:
                argument_list.append([-1, -1, ''])
            else:
                for arg in element['arguments']:
                    arg_start = word2pieces[arg['start']][0]
                    arg_end = word2pieces[arg['end']-1][-1] + 1
                    arg_text = arg['text']
                    if self.config.lang == 'zh':
                        assert self.check_text(''.join(news[arg_start:arg_end]), arg_text)
                    else:
                        assert self.check_text(''.join(news[arg_start:arg_end]), ''.join(arg_text))  
                    argument_list.append([arg_start, arg_end, arg_text])
            
            # every opinion form a quadruple
            if element['opinions'] == []:
                data['quadruples'].append(
                    [[trigger_start, trigger_end, trigger_text], argument_list, [-1, -1, ''], '']
                )
            else:
                for opinion in element['opinions']:
                    opinion_start = word2pieces[opinion['opinion_start']][0]
                    opinion_end = word2pieces[opinion['opinion_end']-1][-1] + 1
                    opinion_text = opinion['opinion_text']
                    sentiment = opinion['sentiment']
                    if self.config.lang == 'zh':
                        assert self.check_text(''.join(news[opinion_start:opinion_end]), opinion_text)
                    else:   
                        assert self.check_text(''.join(news[opinion_start:opinion_end]), ''.join(opinion_text))

                    data['quadruples'].append(
                        [[trigger_start, trigger_end, trigger_text], argument_list, 
                        [opinion_start, opinion_end, opinion_text], sentiment]
                    )

        return data

    
    def remove_duplicate(self, data):
        """
        Remove duplicate elements in a list.
        """

        res = []
        for item in data:
            if item not in res:
                res.append(item)
        return res
    

    def check_text(self, tokenized_text, source_text):
        t0 = tokenized_text.replace('##', '').replace(self.config.unk, '').lower()
        t1 = source_text.replace(' ', '').lower()
        # t1 = source_text.replace('\xa0', '').lower()
        for k in self.config.unkown_tokens:
            t1 = t1.replace(k, '')
        if t0 != t1:
            logger.info(t1 + '||' + t1)
            logger.info(tokenized_text + '||' + source_text)
            raise AssertionError("{} != {}".format(t0, t1))
        return t0 == t1


    def get_dict(self):
        self.polarity_dict = self.wordpair.polarity_dic
        

    def get_pair(self, full_quadruples):
        pairs = {'trigger-arg': set(), 'trigger-opinion': set()}
        for quad in full_quadruples:
            trigger_start, trigger_end, arg_list, opinion_start, opinion_end, sentiment = quad
            if trigger_start != -1:
                for arg in arg_list:
                    arg_start, arg_end = arg
                    if arg_start != -1:
                        pairs['trigger-arg'].add((trigger_start, trigger_end, arg_start, arg_end))
                    # if opinion_start != -1:
                    #     pairs['arg-opinion'].add((arg_start, arg_end, opinion_start, opinion_end))
                if opinion_start != -1:
                    pairs['trigger-opinion'].add((trigger_start, trigger_end, opinion_start, opinion_end))
                
        return pairs

    def find_utterance_index(self, sentence_lengths):
        replies = [i for i in range(len(sentence_lengths))]
        sentence_index = [w for w in replies]

        utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]
        utterance_index = [w for line in utterance_index for w in line]

        token_index = [list(range(sentence_lengths[0]))]
        lens = len(token_index[0])
        for i, w in enumerate(sentence_lengths):
            if i == 0: continue
            if sentence_index[i] == 1:
                distance = lens
            token_index += [list(range(distance, distance + w))]
            distance += w
        token_index = [w for line in token_index for w in line]

        return utterance_index, token_index, sentence_lengths

    def transform2indices(self, dataset, mode='train'):
        res = []
        for document in dataset:
            sentences, pieces2words = document['sentences'], document['pieces2words']
            if mode == 'train':
                triggers, arguments, opinions, quadruples = document['triggers'], document['arguments'], document['opinions'], document['quadruples']
        
            doc_id = document['doc_id']

            sentence_length = list(map(lambda x: len(x) + 2, sentences))

            token2sentenceId = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentenceId = [w for line in token2sentenceId for w in line]

            # New token indices with CLS and SEP to old token indices without CLS and SEP
            new2old = {}
            cur_len = 0
            for i in range(len(sentence_length)):
                for j in range(sentence_length[i]):
                    if j == 0 or j == sentence_length[i] - 1:
                        new2old[len(new2old)] = -1
                    else:
                        new2old[len(new2old)] = cur_len
                        cur_len += 1
            tokens = [[self.config.cls] + sentence + [self.config.sep] for sentence in sentences]

            new_sentences_ids = [[i] * len(w) for i, w in enumerate(tokens)]
            new_sentences_ids = [w for line in new_sentences_ids for w in line]

            flatten_tokens = [w for line in tokens for w in line]
            sentence_end = [i - 1 for i, w in enumerate(flatten_tokens) if w == self.config.sep]
            sentence_start = [i + 1 for i, w in enumerate(flatten_tokens) if w == self.config.cls]
            
            utterance_spans = list(zip(sentence_start, sentence_end))
            utterance_index, token_index, thread_length= self.find_utterance_index(sentence_length)

            # bert input args
            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]

            new_quadruples, pairs, entity_lists, relation_lists, polarity_lists = [], [], [], [], []
            full_quadruples = []
            if mode == 'train':
                triggers = [(s + 2 * token2sentenceId[s] + 1, e + 2 * token2sentenceId[s]) for s, e, t in triggers]
                arguments = [(s + 2 * token2sentenceId[s] + 1, e + 2 * token2sentenceId[s]) for s, e, t in arguments]
                opinions = [(s + 2 * token2sentenceId[s] + 1, e + 2 * token2sentenceId[s], p) for s, e, t, p in opinions]
                opinions = list(set(opinions))

                # quadruple 
                for trigger, args, opinion, sentiment in quadruples:
                    new_index = lambda start, end : (-1, -1) if start == -1 else (start + 2 * token2sentenceId[start] + 1, end + 2 * token2sentenceId[start])
                    trigger_start, trigger_end = new_index(trigger[0], trigger[1])
                    arg_starts, arg_ends = zip(*[new_index(arg[0], arg[1]) for arg in args])
                    opinion_start, opinion_end = new_index(opinion[0], opinion[1])
                    # combine arg to a list
                    arg_list = list(zip(arg_starts, arg_ends))
                    line = (trigger_start, trigger_end, arg_list, opinion_start, opinion_end, 0 if sentiment == '' else self.polarity_dict[sentiment])
                    full_quadruples.append(line)
                    if all(w != -1 for w in [trigger_start, opinion_start]):
                        new_quadruples.append(line)
                    
                relation_lists = self.wordpair.encode_relation(full_quadruples)
                pairs = self.get_pair(full_quadruples)

                # [start, end, entity_type]
                trigger_lists = self.wordpair.encode_entity(triggers, 'ENT-Tri')
                argument_lists = self.wordpair.encode_entity(arguments, 'ENT-Arg')
                opinion_lists = self.wordpair.encode_entity(opinions, 'ENT-Opi')

                entity_lists = trigger_lists + argument_lists + opinion_lists

                polarity_lists = self.wordpair.encode_sentiment(new_quadruples)

            # new_sentences_ids is equal to utterance_index because every document is independent
            res.append((doc_id, input_ids, input_masks, input_segments, sentence_length, 
                            new_sentences_ids, utterance_index, token_index, thread_length,
                            pieces2words, new2old, full_quadruples, 
                            pairs, entity_lists, relation_lists, polarity_lists))

        return res

    def repack_unknow(self, source_sequence):
        '''
        # sentence='🍎12💩', Bert can't recognize two contiguous emojis, so it recognizes the whole as '[UNK]'
        # We need to manually split it, recognize the words that are not in the bert vocabulary as UNK, 
        and let BERT re-segment the parts that can be recognized, such as numbers
        # The above example processing result is: ['[UNK]', '12', '[UNK]']
        '''
        if len(source_sequence) > 1:
            return source_sequence
        lst = list(re.finditer('|'.join(' 🍔—🐛🙉🙄🔨🏆🆔👌👀🥺冖🌚🙈😭🍎😅💩尛硌糇💰🐴🙊💯⭐🐶🐟🙏😄🏻📶🐮🍺❌🤔🐍🐸🙃🤣🏆😂🌚“”簋讣埈髌碜慜郯嚯磙诹亖蓥'), source_sequence))
        if not lst:
            return source_sequence
        return '[UNK]'
    
    def forward(self):
        modes = ['train', 'valid', 'test']
        datasets = {}

        for mode in modes:
            data = self.read_data(mode)
            datasets[mode] = data
        
        label_dict = self.get_dict()

        res = {}
        for mode in modes:
            res[mode] = self.transform2indices(datasets[mode], mode)

        res['label_dict'] = label_dict
        return res
    