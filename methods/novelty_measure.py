from distutils.command.config import config
import os
import sys
from logging import getLogger
from time import time
import datetime
import argparse
import yaml
import pickle
import json

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from metrics import *




def get_config(config_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--foldk', type=int, default=0)
    parser.add_argument('--filter_repeat', type=int, default=0, help='determine how much repeat samples we will use during training.')
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--method', type=str)
    parser.add_argument('--loss_type', type=str, default='CE', help='BPR or CE')
    args, _ = parser.parse_known_args()
    input_config = args.__dict__

    dataset_config_path = f"../data/{input_config['dataset']}/data_config.yaml"
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.load(f, Loader=yaml.FullLoader)
    model_config_path = f"{input_config['method']}/model_config.yaml"
    with open(model_config_path, 'rb') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    config = {**dataset_config, **model_config, **input_config}

    config['foldk_path'] = f"../data/{input_config['dataset']}/foldk.pkl"
    config['data_path'] = f"../data/{input_config['dataset']}/{input_config['dataset']}-filter-new.pkl"

    config['model_folder'] = f"{config['method']}/model/{config['dataset']}/drop_rep_{config['filter_repeat']}/{config['foldk']}/"
    config['eval_folder'] = f"{config['method']}/eval/{config['dataset']}/drop_rep_{config['filter_repeat']}/{config['foldk']}/"
    config['exposure_folder'] = f"{config['method']}/exposure/{config['dataset']}/drop_rep_{config['filter_repeat']}/{config['foldk']}/"

    config['prediction_folder'] = f"{config['method']}/pred/{config['dataset']}/drop_rep_{config['filter_repeat']}/{config['foldk']}/"
    # some require config['device']
    if not os.path.exists(config['exposure_folder']):
        os.makedirs(config['exposure_folder'])
    
    return config

def label_merit_distribution(data, user_list):
    item_label_dict = {'overall': dict(), 'rep': dict(), 'expl': dict()}
    for uid in user_list:
        user_seq = data[uid]
        pos_item = user_seq[-1]
        if pos_item in item_label_dict['overall'].keys():
            item_label_dict['overall'][pos_item] += 1
        else:
            item_label_dict['overall'][pos_item] = 1

        if pos_item in user_seq[:-1]: #whether it is a rep label
            if pos_item in item_label_dict['rep'].keys():
                item_label_dict['rep'][pos_item] += 1
            else:
                item_label_dict['rep'][pos_item] = 1
        else:
            if pos_item in item_label_dict['expl'].keys():
                item_label_dict['expl'][pos_item] += 1
            else:
                item_label_dict['expl'][pos_item] = 1
    return item_label_dict

def get_rep_expl_stats(metric_dict, item):
    
    overall = metric_dict['overall'][item] if item in metric_dict['overall'].keys() else 0
    rep = metric_dict['rep'][item] if item in metric_dict['rep'].keys() else 0
    expl = metric_dict['expl'][item] if item in metric_dict['expl'].keys() else 0
    rep_ratio = rep/overall if overall != 0 else 0
    return overall, rep, expl, rep_ratio

def label_merit_exposure_analysis(item_list, label_dict, merit_dict, exposure_obj):
    expo1_dict, expo5_dict, expo20_dict = exposure_obj.get_exposure()
    analysis_dict = dict()
    for item in item_list:
        expo1_overall, expo1_rep, expo1_expl, expo1_rep_ratio = get_rep_expl_stats(expo1_dict, item)
        expo5_overall, expo5_rep, expo5_expl, expo5_rep_ratio = get_rep_expl_stats(expo5_dict, item)
        label_overall, label_rep, label_expl, label_rep_ratio = get_rep_expl_stats(label_dict, item)
        merit_overall, merit_rep, merit_expl, merit_rep_ratio = get_rep_expl_stats(merit_dict, item)
        analysis_dict[item] = {'rep_ratio': [expo1_rep_ratio, expo5_rep_ratio, merit_rep_ratio, label_rep_ratio],
                                'rep': [expo1_rep, expo5_rep, merit_rep, label_rep],
                                'expl': [expo1_expl, expo5_expl, merit_expl, label_expl],
                                'overall': [expo1_overall, expo5_overall, merit_overall, label_overall]}
    return analysis_dict



if __name__ == '__main__':
    config = get_config()

    with open(config['data_path'], 'rb') as f:
        data = pickle.load(f)
    with open(config['foldk_path'], 'rb') as f:
        foldk = pickle.load(f)
    with open(config['prediction_folder'] + "best_overall_prediction.pkl", 'rb') as f:
        pred = pickle.load(f)
    
    test_users = foldk[config['foldk']]['test']
    train_users = foldk[config['foldk']]['train']

    # label distribution
    item_label_dict = label_merit_distribution(data, train_users)
    
    # calculate merit
    item_merit_dict = label_merit_distribution(data, test_users)

    # find items you want to know
    sorted_label = sorted(item_label_dict['overall'].items(), key=lambda x:x[1], reverse=True)
    sorted_merit = sorted(item_merit_dict['overall'].items(), key=lambda x:x[1], reverse=True)
    topk_merit_item = [item for item, merit in sorted_merit[:100]]
    topk_label_item = [item for item, merit in sorted_label[:100]]

    log_novelty = Log_novelty()
    for uid in test_users:
        user_seq = data[uid]
        user_pred = pred[uid]
        item_seq = user_seq[:-1]
        pos_item = user_seq[-1]
        log_novelty.update(user_pred, item_seq, pos_item, 1)
        log_novelty.update(user_pred, item_seq, pos_item, 5)
        log_novelty.update(user_pred, item_seq, pos_item, 10)
    all_n1, rep_n1, expl_n1 = log_novelty.get_average(1)
    all_n5, rep_n5, expl_n5 = log_novelty.get_average(5)
    all_n10, rep_n10, expl_n10 = log_novelty.get_average(10)
    print("Novelty@1== all: {} \t rep: {} \t expl: {}".format(all_n1, rep_n1, expl_n1))
    print("Novelty@5== all: {} \t rep: {} \t expl: {}".format(all_n5, rep_n5, expl_n5))
    print("Novelty@10== all: {} \t rep: {} \t expl: {}".format(all_n10, rep_n10, expl_n10))


        
        
        


    


