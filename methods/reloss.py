from cmath import inf
from distutils.command.build import build
from distutils.command.config import config
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

class RELoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, config):
        super(RELoss, self).__init__()
        self.config = config
        self.device = config['device']
        self.loss_fct = F.nll_loss
        self.build_candidate(config)
    
    def build_candidate(self, config):
        self.pre_trained_model = Word2Vec.load(config['pretrain_path'])
        self.pre_trained_item_set = self.pre_trained_model.wv.index_to_key
        with open(config['toppop_path'], 'rb') as f:
            pop_topk = pickle.load(f)
        self.pop_top = pop_topk[config['dataset']]
    
    def get_candidate(self, item_list, topm):
        positive_list = [i for i in item_list if i in self.pre_trained_item_set]
        # print(self.pre_trained_model.wv.most_similar(positive=positive_list))
        if len(positive_list) != 0:
            cands = self.pre_trained_model.wv.most_similar(positive=positive_list)
            item_set = [cand[0] for cand in cands if cand[1]>0.5]
        else:
            item_set = set()
        expl_cand = set(item_set)|set(self.pop_top[:topm])-set(item_list)
        return list(expl_cand)

    def forward(self, pred, pos_items, item_seq):
        batch_size = item_seq.size(0)
        pred_list = []
        for pred_n, pos_items_n, item_seq_n in zip(pred, pos_items, item_seq):
            hist_item_set = set(item_seq_n.tolist())
            # hist_item_set.remove(0)
            pos_items_i = pos_items_n.item()
            if pos_items_i in hist_item_set: # mask explore user, since it is repeat mode now
                rep_mask = torch.ones_like(pred_n).to(self.device)
                rep_mask[list(hist_item_set)] = 0.0
                pred_n = pred_n + rep_mask*float(-1e9)
                pred_n = torch.log(F.softmax(pred_n) + 1e-8)
                pred_list.append(pred_n)
                # print('repeat')
            else:
                # explore loss. mask rep items and 
                # print('explore')
                if self.config['reduce_expl_space'] == 1:
                    expl_mask = torch.ones_like(pred_n).to(self.device)
                    cand_items = self.get_candidate(hist_item_set, 500)
                    expl_mask[cand_items] = 0.0
                else:
                    expl_mask = torch.zeros_like(pred_n).to(self.device)
                    expl_mask[list(hist_item_set)] = 1
                pred_n = pred_n + expl_mask*float(-1e9)
                pred_n = torch.log(F.softmax(pred_n) + 1e-8)
                pred_list.append(pred_n)
        final_pred = torch.stack(pred_list, axis=0)
        loss = self.loss_fct(final_pred, pos_items)
        # print(loss)
        return loss
