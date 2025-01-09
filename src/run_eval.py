import argparse
import json
import numpy as np
from collections import OrderedDict

class Run_eval:
    def __init__(self, config):
        self.pred_file = config.pred_file
        self.gold_file = config.gold_file
    
    def remove_duplicate(self, data):
        """
        Remove duplicate elements in a list.
        """

        res = []
        for item in data:
            if item not in res:
                res.append(item)
        return res
    
    def read_gold_data(self, path, mode='pred', lang='zh'):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            content = {doc['doc_id']: doc for doc in content}


        new_content = {}
        for id, doc in content.items():
            quadruples = []
            tri_opi_senti_pairs = []
            tri_opi = []
            tri_arg = []
            # include null element in the quadruples because some triggers don't have any arguments or opinions
            event_mentions = doc['event_mentions']
            # build quadruples
            for element in event_mentions:
                trigger_text = element['trigger']['text']
                trigger_start = element['trigger']['start']
                trigger_end = element['trigger']['end']

                argument_list = []
                if element['arguments'] == []:
                    argument_list.append([-1, -1, ''])
                else:
                    for argument in element['arguments']:
                        argument_start = argument['start']
                        argument_end = argument['end']
                        argument_text = argument['text']
                        if lang == 'zh':
                            argument_list.append([argument_start, argument_end, argument_text])
                            tri_arg.append([trigger_start, trigger_end, argument_start, argument_end, trigger_text, argument_text])
                        elif lang == 'en':
                            argument_list.append([argument_start, argument_end, ' '.join(argument_text)])
                            tri_arg.append([trigger_start, trigger_end, argument_start, argument_end, ' '.join(trigger_text), ' '.join(argument_text)])

                if element['opinions'] == []:
                    if lang == 'zh':
                        quadruples.append(
                            [trigger_start, trigger_end, trigger_text, argument_list, -1, -1, '', '']
                        )
                    elif lang == 'en':
                        quadruples.append(
                            [trigger_start, trigger_end, ' '.join(trigger_text), argument_list, -1, -1, '', '']
                        )
                else:
                    for opinion in element['opinions']:
                        opinion_text = opinion['opinion_text']
                        opinion_start = opinion['opinion_start']
                        opinion_end = opinion['opinion_end']
                        sentiment = opinion['sentiment']
                        if lang == 'zh':
                            quadruples.append(
                                [trigger_start, trigger_end, trigger_text, argument_list, opinion_start, opinion_end, opinion_text, sentiment]
                            )
                            tri_opi.append([trigger_start, trigger_end, opinion_start, opinion_end, trigger_text, opinion_text])
                            tri_opi_senti_pairs.append([trigger_start, trigger_end, opinion_start, opinion_end, sentiment, trigger_text, opinion_text])
                        elif lang == 'en':
                            quadruples.append(
                                [trigger_start, trigger_end, ' '.join(trigger_text), argument_list, opinion_start, opinion_end, ' '.join(opinion_text), sentiment]
                            )   
                            tri_opi.append([trigger_start, trigger_end, opinion_start, opinion_end, ' '.join(trigger_text), ' '.join(opinion_text)])
                            tri_opi_senti_pairs.append([trigger_start, trigger_end, opinion_start, opinion_end, sentiment, ' '.join(trigger_text), ' '.join(opinion_text)])
        
            # build trigger, arguments, opinion list
            if lang == 'zh':
                trigger_list = [[event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['trigger']['text']] for event_mention in event_mentions]
                opinion_list = [[opinion['start'], opinion['end'], opinion['text']] for opinion in doc['opinions']]

            elif lang == 'en':
                trigger_list = [[event_mention['trigger']['start'], event_mention['trigger']['end'], ' '.join(event_mention['trigger']['text'])] for event_mention in event_mentions]
                opinion_list = [[opinion['start'], opinion['end'], ' '.join(opinion['text'])] for opinion in doc['opinions']]

            arguments = [event_mention['arguments'] for event_mention in event_mentions]
            argument_list = []
            for argument in arguments:
                for arg in argument:
                    # argument_list.append([arg['start'], arg['end']])
                    if lang == 'zh':
                        argument_list.append([arg['start'], arg['end'], arg['text']])
                    elif lang == 'en':
                        argument_list.append([arg['start'], arg['end'], ' '.join(arg['text'])])
            
            trigger_list = self.remove_duplicate(trigger_list)
            argument_list = self.remove_duplicate(argument_list)
            opinion_list = self.remove_duplicate(opinion_list)


            pack = {
                'doc_id': doc['doc_id'],
                'triggers': trigger_list,
                'arguments': argument_list,
                'opinions': opinion_list,
                'trigger-arg': tri_arg,
                'trigger-opinion': tri_opi,
                'full_quadruples': quadruples,
                'tri_opi_senti_pairs': tri_opi_senti_pairs
            }
            new_content[id] = pack

        return new_content

    def read_pred_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            content = {doc['doc_id']: doc for doc in content}
        return content
    
    def post_process(self, line, key='quad'):
        if key in ['triggers', 'arguments', 'opinions']:
            return [tuple(w[:2]) for w in line[key]]
        if key in ['trigger-arg', 'trigger-opinion']:
            return [tuple(w[:4]) for w in line[key]]
        if key == 'tri_opi_senti_pairs':
            return [tuple(w[:5]) for w in line[key]]
        
        res = []
        if key in ['quad', 'iden']:
            for quad in line['full_quadruples']:
                trigger_start, trigger_end = quad[:2]
                arg_list = quad[3]
                opinion_start, opinion_end = quad[4:6]
                # flatten the argument list
                arg_list = [tuple(arg[:2]) for arg in arg_list]
                sentiment = quad[-1]
                if key == 'quad':
                    res.append(tuple([trigger_start, trigger_end, arg_list, opinion_start, opinion_end, sentiment]))
                else:
                    res.append(tuple([trigger_start, trigger_end, arg_list, opinion_start, opinion_end]))
        return res
                

    def post_process_for_text(self, line, key='triggers'):
        if key in ['triggers', 'arguments', 'opinions']:
            res = [tuple(w[-1]) for w in line[key]]
            return list(OrderedDict.fromkeys(res))
        if key in ['trigger-arg', 'trigger-opinion']:
            res = [tuple(w[4:]) for w in line[key]]
            return list(OrderedDict.fromkeys(res))
        if key == 'tri_opi_senti_pairs':
            res = [tuple(w[4:]) for w in line[key]]
            return list(OrderedDict.fromkeys(res))
    
    def post_process_for_english_text(self, line, key='triggers'):
        if key in ['triggers', 'arguments', 'opinions']:
            # res = [tuple([' '.join(w[-1])]) for w in line[key]]
            res = [tuple([w[-1]]) for w in line[key]]
            return list(OrderedDict.fromkeys(res))
        if key in ['trigger-arg', 'trigger-opinion']:
            # res = [tuple([' '.join(w[4:])]) for w in line[key]]
            # res = [tuple([' '.join(w[-2]), ' '.join(w[-1])]) for w in line[key]]
            res = [tuple([w[-2], w[-1]]) for w in line[key]]
            return list(OrderedDict.fromkeys(res))
        if key == 'tri_opi_senti_pairs':
            # res = [tuple([' '.join(w[4:])]) for w in line[key]]
            # res = [tuple([' '.join(w[-3]), ' '.join(w[-2]), ' '.join(w[-1])]) for w in line[key]]
            res = [tuple([w[-3], w[-2], w[-1]]) for w in line[key]]
            return list(OrderedDict.fromkeys(res))

    
    def compute_score_for_text(self, mode='triggers', lang='zh'):
        tp, fp, fn = 0, 0, 0
        for doc_id in self.gold_res:
            pred_line = self.pred_res[doc_id]
            gold_line = self.gold_res[doc_id]

            if lang == 'zh':
                pred_line = self.post_process_for_text(pred_line, mode)
                gold_line = self.post_process_for_text(gold_line, mode)
            elif lang == 'en':
                pred_line = self.post_process_for_english_text(pred_line, mode)
                gold_line = self.post_process_for_english_text(gold_line, mode)

            # compare tuple list, list format is [('text'), ...] or [('text', ...), ...],
            # every tuple is ordered
            fp += len(set(pred_line) - set(gold_line))  
            fn += len(set(gold_line) - set(pred_line))
            tp += len(set(pred_line) & set(gold_line))
        
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        scores = [p, r, f1]
        return scores

    def compute_partial_matching_score_for_text(self, mode='triggers', lang='zh', alpha=0.5):
        tp, fp, fn = 0, 0, 0
        res = []
        for doc_id in self.gold_res:
            tmp_tp = 0
            gold_line = self.gold_res[doc_id]
            pred_line = self.pred_res[doc_id]
            if lang == 'zh':
                pred_line = self.post_process_for_text(pred_line, mode)
                gold_line = self.post_process_for_text(gold_line, mode)
            elif lang == 'en':
                pred_line = self.post_process_for_english_text(pred_line, mode)
                gold_line = self.post_process_for_english_text(gold_line, mode)
            
            if mode in ['triggers', 'arguments', 'opinions']:
                for pred in pred_line:
                    tp_p = 0
                    if lang == 'en':
                        pred = pred[0].split()
                    for gold in gold_line:
                        if lang == 'zh':
                            matching_chars = self.count_matching_chars(pred, gold)
                        elif lang == 'en':
                            gold = gold[0].split()
                            matching_chars = self.count_matching_chars(pred, gold)
                        if matching_chars >= alpha * len(gold):
                            tp_p = 1
                            break
                    tmp_tp += tp_p
            elif mode in ['trigger-arg', 'trigger-opinion']:
                for pred in pred_line:
                    tp_p = 0
                    pred_trigger, pred_else = pred
                    for gold in gold_line:
                        gold_trigger, gold_else = gold
                        if lang == 'zh':
                            trigger_matching_chars = self.count_matching_chars(pred_trigger, gold_trigger)
                            else_matching_chars = self.count_matching_chars(pred_else, gold_else)
                            if trigger_matching_chars > alpha * len(gold_trigger) and else_matching_chars >= alpha * len(gold_else):
                                tp_p = 1
                                break
                        elif lang == 'en':
                            gold_trigger = gold_trigger.split()
                            gold_else = gold_else.split()
                            trigger_matching_chars = self.count_matching_chars(pred_trigger.split(), gold_trigger)
                            else_matching_chars = self.count_matching_chars(pred_else.split(), gold_else)
                            if trigger_matching_chars >= alpha * len(gold_trigger) and else_matching_chars >= alpha * len(gold_else):
                                tp_p = 1
                                break
                    tmp_tp += tp_p
            else:
                for pred in pred_line:
                    tp_p = 0
                    pred_sentiment, pred_trigger, pred_opi = pred
                    for gold in gold_line:
                        gold_sentiment, gold_trigger, gold_opi = gold
                        sentiment_matching_chars = 1 if pred_sentiment == gold_sentiment else 0
                        if lang == 'zh':
                            trigger_matching_chars = self.count_matching_chars(pred_trigger, gold_trigger)
                            else_matching_chars = self.count_matching_chars(pred_opi, gold_opi)
                            if trigger_matching_chars >= alpha * len(gold_trigger) and else_matching_chars >= alpha * len(gold_opi) and sentiment_matching_chars == 1:
                                tp_p = 1
                                break
                        elif lang == 'en':
                            gold_trigger = gold_trigger.split()
                            gold_opi = gold_opi.split()
                            trigger_matching_chars = self.count_matching_chars(pred_trigger.split(), gold_trigger)
                            else_matching_chars = self.count_matching_chars(pred_opi.split(), gold_opi)
                            if trigger_matching_chars >= alpha * len(gold_trigger) and else_matching_chars >= alpha * len(gold_opi) and sentiment_matching_chars == 1:
                                tp_p = 1
                                break
                    tmp_tp += tp_p
            tp += tmp_tp
            fp += len(pred_line) - tmp_tp
            fn += len(gold_line) - tmp_tp
        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        scores = [p, r, f1]
        return scores

    def count_matching_chars(self, pred, gold):
        # we use Longest Common Substring to calculate the matching characters
        pred_len = len(pred)
        gold_len = len(gold)
        dp = np.zeros((pred_len + 1, gold_len + 1))
        max_len = 0  # variable to store the length of longest common substring
        for i in range(1, pred_len + 1):
            for j in range(1, gold_len + 1):
                if pred[i - 1] == gold[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_len = max(max_len, dp[i][j])
                else:
                    dp[i][j] = 0
        return max_len


    def forward(self, print_line=False, lang='zh'):
        self.pred_res = self.read_pred_data(self.pred_file)
        self.gold_res = self.read_gold_data(self.gold_file, lang=lang)

        assert len(self.pred_res) == len(self.gold_res) 
        
        # calculate the metrics
        scores = []
        res = 'Item\t\tPrec.\tRec.\tF1\tPPrec.\tPRec.\tPF1\n'
        items = ['triggers', 'arguments', 'opinions', 'trigger-arg', 'trigger-opinion', 'tri_opi_senti_pairs']
        item_name = ['Triggers', 'Arguments', 'Opinions', 'Tri-Arg', 'Tri-Opi', 'Tri-Opi-Senti']
        num_format = lambda x: '\t\t' + '\t'.join([f'{w*100:.2f}' if i < 6 else str(w) for i, w in enumerate(x)]) + '\n'
        line_indeces = [0, 3, 6]
        for i, item in enumerate(items):
            if i in line_indeces: res += '-'*30 + '\n'
            score = self.compute_score_for_text(item, lang)
            partial_score = self.compute_partial_matching_score_for_text(item, lang)
            score.extend(partial_score)
            scores.append(score)

            res += item_name[i] + num_format(score)
        
        tri_arg_score, tri_opi_senti_score = scores[3], scores[5]
        return tri_arg_score, tri_opi_senti_score, res