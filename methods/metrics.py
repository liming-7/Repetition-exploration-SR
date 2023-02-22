import math
import numpy as np

# this is for only one ground truth.
def ndcg(ranks, label, k):

    if label in ranks[:k]:
        label_rank = ranks.index(label)
        return 1.0/math.log2(label_rank + 2)
    else:
        return 0

def recall(ranks, label, k):
    if label in ranks[:k]:
        return 1
    else:
        return 0

def mrr(ranks, label, k):
    if label in ranks[:k]:
        label_rank = ranks.index(label)
        return 1.0/(label_rank+1)
    else:
        return 0

def exposure(ranks, k):
    exp_dict = dict()
    for item in ranks[:k]:
        exp_dict[item] = 1
    return exp_dict


def log_exposure(ranks, k):
    exp_dict = dict()
    rank = 1.0
    for item in ranks[:k]:
        exp_dict[item] = 1/math.log2(rank+1)
        rank += 1
    return exp_dict

# the novelty is a ratio.
def get_novelty(ranks, item_seq, k):
    total = 0.0
    novel = 0.0
    rank = 1.0
    for item in ranks[:k]:
        if item not in item_seq:
            novel += 1/math.log2(k+1)
        total += 1/math.log2(k+1)
        rank += 1
    return novel/total

class Log_novelty(object):

    def __init__(self, topk_list) -> None:
        self.novelty_dict = dict()
        # according to user
        for topk in topk_list:
            self.novelty_dict[topk] = {'overall': [], 'rep': [], 'expl': []}
        # self.novelty_dict[5] = {'overall': [], 'rep': [], 'expl': []}
        # self.novelty_dict[10] = {'overall': [], 'rep': [], 'expl': []}
    
    # compute novelty on different users
    def update(self, ranks, item_seq, pos_item, k):
        novelty = get_novelty(ranks, item_seq, k)
        self.novelty_dict[k]['overall'].append(novelty)
        if pos_item in item_seq:
            self.novelty_dict[k]['rep'].append(novelty)
        else:
            self.novelty_dict[k]['expl'].append(novelty)
    
    def get_novelty(self, topk):
        return self.novelty_dict[topk]

class Log_exposure(object):
    def __init__(self, topk_list):
        self.item_expo_dict = dict()
        self.expo_dict = dict()
        for topk in topk_list:
            self.item_expo_dict[topk] = {'overall': dict(), 'rep': dict(), 'expl': dict()}
            self.expo_dict[topk]= {'rep':0.0, 'expl':0.0}

        # self.item_expo_dict[5] = {'overall': dict(), 'rep': dict(), 'expl': dict()}
        # self.item_expo_dict[10] = {'overall': dict(), 'rep': dict(), 'expl': dict()}

    def update(self, ranks, item_seq, k):
        rank = 1.0
        for item in ranks[:k]:
            # print (self.item_expo_dict[k]['overall'])
            if item in self.item_expo_dict[k]['overall'].keys():
                self.item_expo_dict[k]['overall'][item] += 1/math.log2(rank+1)
            else:
                self.item_expo_dict[k]['overall'][item] = 1/math.log2(rank+1)
            # compute repeat exposure and fresh exposure of the item.
            if item in item_seq:
                self.expo_dict[k]['rep'] += 1/math.log2(rank+1)
                if item in self.item_expo_dict[k]['rep'].keys():
                    self.item_expo_dict[k]['rep'][item] += 1/math.log2(rank+1)
                else:
                    self.item_expo_dict[k]['rep'][item] = 1/math.log2(rank+1)
            else:
                self.expo_dict[k]['expl'] += 1/math.log2(rank+1)

                if item in self.item_expo_dict[k]['expl'].keys():
                    self.item_expo_dict[k]['expl'][item] += 1/math.log2(rank+1)
                else:
                    self.item_expo_dict[k]['expl'][item] = 1/math.log2(rank+1)
            rank +=1
    
    def get_exposure(self, topk):
        return self.item_expo_dict[topk], self.expo_dict[topk]
        # , self.item_expo_dict[5], self.item_expo_dict[10]
        

def re_log_exposure(ranks, item_seq, exp_dict, k):
    rank = 1
    for item in ranks[:k]:
        if item in exp_dict['overall'].keys():
            exp_dict['overall'] += 1/math.log2(rank+1)
        else:
            exp_dict['overall'] = 1/math.log2(rank+1)
        if item in item_seq:
            if item in exp_dict['rep'].keys():
                exp_dict['rep'][item] += 1/math.log2(rank+1)
            else:
                exp_dict['rep'][item] = 1/math.log2(rank+1)
        else:
            if item in exp_dict['expl'].keys():
                exp_dict['expl'][item] += 1/math.log2(rank+1)
            else:
                exp_dict['expl'][item] = 1/math.log2(rank+1)
        rank +=1