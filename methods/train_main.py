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

# import models
from gru4rec.gru4rec import GRU4Rec
from sasrec.sasrec import SASRec
from caser.caser import Caser
from srgnn.srgnn import SRGNN
from bert4rec.bert4rec import BERT4Rec
from repeatnet.repeatnet import RepeatNet
from data_loader_temp import SeqDataset, get_dataloader
from utils import convert_to_gpu, convert_all_data_to_gpu
import itertools
import metrics
import wandb


# caser can not be used in our case, since it learns a user embedding.

def get_config(config_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--foldk', type=int, default=0)
    parser.add_argument('--train_mode', type=int, default=0, help='0: use all data; 1: only use explore labels')
    parser.add_argument('--max_seq_length', type=int, default=50)
    parser.add_argument('--method', type=str)
    parser.add_argument('--loss_type', type=str, default='CE', help='BPR or CE or RE')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--shared_emb', type=int, default=1, help='1 means shared embedding, 0 means independt embedding.')
    # parser.add_argument('--reduce_expl_space', type=int, default=1, help='1: use expl reduce technique, 0: do not use')

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
    config['toppop_path'] = f"../data/global_top.pkl"
    # config['model_folder'] = f"{config['method']}/model/{config['dataset']}/{config['loss_type']}/drop_rep_{config['filter_repeat']}/{config['foldk']}/"
    config['eval_folder'] = f"{config['method']}/eval/{config['dataset']}/{config['loss_type']}/mode{config['train_mode']}_shared{config['shared_emb']}/{config['foldk']}/"
    config['prediction_folder'] = f"{config['method']}/pred/{config['dataset']}/{config['loss_type']}/mode{config['train_mode']}_shared{config['shared_emb']}/{config['foldk']}/"

    # if not os.path.exists(config['model_folder']):
    #     os.makedirs(config['model_folder'])
    if not os.path.exists(config['eval_folder']):
        os.makedirs(config['eval_folder'])
    if not os.path.exists(config['prediction_folder']):
        os.makedirs(config['prediction_folder'])

    if torch.cuda.is_available():
        print("CUDA!!")
        config['device'] = torch.device('cuda:0')
    else:
        config['device'] = torch.device('cpu')
        print("CPU!!")
    # some require config['device']
    sys.stdout.flush()
    return config

# def avg_performance(metric, topk):
    # pass

def eval_acc(topk_list, user_list, item_seq_list, predict_list, label_list):
    # 
    acc_dict = {acc+str(topk):[] for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}
    user_dict = {'user': [], 'rep_user':[], 'expl_user':[]}
    for user, item_seq, predict, label in zip(user_list, item_seq_list, predict_list, label_list):
        user_dict['user'].append(user)
        if label in item_seq:
            user_dict['rep_user'].append(user)
        else:
            user_dict['expl_user'].append(user)
        for topk in topk_list:
            acc_dict['recall'+str(topk)].append(metrics.recall(predict, label, topk))
            acc_dict['ndcg'+str(topk)].append(metrics.ndcg(predict, label, topk))
            acc_dict['mrr'+str(topk)].append(metrics.mrr(predict, label, topk))
    avg_acc_dict = {metric: np.mean(result) for metric, result in acc_dict.items()}
    # get the rep user and expl user
    rep_user_ind = [i for i in range(len(user_dict['user'])) if user_dict['user'][i] in user_dict['rep_user']]
    expl_user_ind = [i for i in range(len(user_dict['user'])) if user_dict['user'][i] in user_dict['expl_user']]
    avg_rep_acc_dict = {metric: np.mean([result[ind] for ind in rep_user_ind]) for metric, result in acc_dict.items()}
    avg_expl_acc_dict = {metric: np.mean([result[ind] for ind in expl_user_ind]) for metric, result in acc_dict.items()}
    return avg_acc_dict, avg_rep_acc_dict, avg_expl_acc_dict
    

def label_merit_distribution(hist_dict, pos_dict):
    item_label_dict = {'overall': dict(), 'rep': dict(), 'expl': dict()}

    total_rep = 0.0
    total_expl = 0.0
    for uid in hist_dict.keys():
        user_seq = hist_dict[uid]
        pos_item = pos_dict[uid]
        if pos_item in item_label_dict['overall'].keys():
            item_label_dict['overall'][pos_item] += 1
        else:
            item_label_dict['overall'][pos_item] = 1.0

        if pos_item in user_seq: #whether it is a rep label
            total_rep += 1
            if pos_item in item_label_dict['rep'].keys():
                item_label_dict['rep'][pos_item] += 1
            else:
                item_label_dict['rep'][pos_item] = 1.0
        else:
            total_expl += 1
            if pos_item in item_label_dict['expl'].keys():
                item_label_dict['expl'][pos_item] += 1
            else:
                item_label_dict['expl'][pos_item] = 1.0
    # micro
    golden_fresh_ratio_dict = dict()
    for item in item_label_dict['overall'].keys():
        if item in item_label_dict['expl'].keys():
            golden_fresh_ratio_dict[item] = item_label_dict['expl'][item]/item_label_dict['overall'][item]
        else:
            golden_fresh_ratio_dict[item] = 0.0
    # macro fr
    wandb.log({'golden_macro_fr': total_expl/(total_expl+total_rep)})
    return item_label_dict, golden_fresh_ratio_dict

def eval_exposure(pred_dict, hist_dict, pos_dict, item_label_dict, golden_fresh_ratio_dict, topk_list, toplabelk_list, config):
    
    # rank item according to freq in 
    sorted_label = sorted(item_label_dict['overall'].items(), key=lambda x:x[1], reverse=True)
    topk_label_item_list = [item for item, merit in sorted_label if merit >= 10]
    # get overall item repeat fresh and repeat exposure across all items.
    # migh be useful for case study.
    log_exposure = metrics.Log_exposure(topk_list)
    for usr in pred_dict.keys():
        user_pred = pred_dict[usr]
        hist_seq = hist_dict[usr]
        pos_item = pos_dict[usr]
        for topk in topk_list:
            log_exposure.update(user_pred, hist_seq, topk)

    # max default set to 1000       
    max_item_list = topk_label_item_list[:1000]

    pred_fresh_ratio_dict = {topk:{} for topk in topk_list}
    for topk in topk_list:
        exposure_dict_topk, expo_dict = log_exposure.get_exposure(topk)
        # log macro fr
        wandb.log({f'pred_macro_fr@{topk}': expo_dict['expl']/(expo_dict['rep']+expo_dict['expl'])})
        for item in max_item_list:
            if item in exposure_dict_topk['overall'].keys(): # remove items do not any expousre at all
                if item in exposure_dict_topk['expl'].keys():
                    pred_fresh_ratio_dict[topk][item] = exposure_dict_topk['expl'][item]/exposure_dict_topk['overall'][item]
                else:
                    pred_fresh_ratio_dict[topk][item] = 0
    
    # compute L1 and L2 across different topk items []
    # avg_fresh_ratio = dict()
    wandb_dict = dict()
    save_dict = dict()
    for toplabelk in toplabelk_list:
        ana_item_list = topk_label_item_list[:toplabelk]
        for topk in topk_list:
            pred_fresh_exposure_ratio_list = []
            golden_fresh_exposure_ratio_list = []
            predvsgolden_fresh_exposure_ratio_list = []
            for item in ana_item_list:
                if item in pred_fresh_ratio_dict[topk].keys():
                    pred_fresh_exposure_ratio_list.append(pred_fresh_ratio_dict[topk][item])
                    golden_fresh_exposure_ratio_list.append(golden_fresh_ratio_dict[item])
                    predvsgolden_fresh_exposure_ratio_list.append(pred_fresh_ratio_dict[topk][item]-golden_fresh_ratio_dict[item])#pred-golden
            wandb_dict[f'freq{toplabelk}_avg_pred_fr@{topk}'] = np.mean(pred_fresh_exposure_ratio_list)
            wandb_dict[f'freq{toplabelk}_avg_golden_fr@{topk}'] = np.mean(golden_fresh_exposure_ratio_list)
            wandb_dict[f'freq{toplabelk}_avg_vs_fr_diff@{topk}'] = np.mean(predvsgolden_fresh_exposure_ratio_list)
            l1_diff_list = [abs(diff) for diff in predvsgolden_fresh_exposure_ratio_list]
            wandb_dict[f'freq{toplabelk}_avg_vs_fr_l1@{topk}'] = np.mean(l1_diff_list)
            
            # save this in case a distribution map is needed.
            save_dict[f'freq{toplabelk}_pred_fr@{topk}'] = pred_fresh_exposure_ratio_list
            save_dict[f'freq{toplabelk}_golden_fr@{topk}'] = golden_fresh_exposure_ratio_list
            save_dict[f'freq{toplabelk}_vs_fr_diff@{topk}'] = predvsgolden_fresh_exposure_ratio_list
    wandb.log(wandb_dict)
    with open(config['eval_folder']+"micro_fr_distribution.pkl", 'wb') as f:
        pickle.dump(save_dict, f)
    return log_exposure
    


def eval_novelty(pred_dict, hist_dict, pos_dict, topk_list):
    
    log_novelty = metrics.Log_novelty(topk_list)
    for usr in pred_dict.keys():
        user_pred = pred_dict[usr]
        hist_seq = hist_dict[usr]
        pos_item = pos_dict[usr]
        for topk in topk_list:
            log_novelty.update(user_pred, hist_seq, pos_item, topk)

    # wandb log
    wandb_dict = dict()
    save_dict = dict()
    for topk in topk_list:
        novelty_dict = log_novelty.get_novelty(topk)
        wandb_dict[f'avg_overall_novelty@{topk}'] = np.mean(novelty_dict['overall'])
        wandb_dict[f'avg_rep_novelty@{topk}'] = np.mean(novelty_dict['rep'])
        wandb_dict[f'avg_expl_novelty@{topk}'] = np.mean(novelty_dict['expl'])
        
        save_dict[f'overall_novelty@{topk}'] = novelty_dict['overall']
        save_dict[f'rep_novelty@{topk}'] = novelty_dict['rep']
        save_dict[f'expl_novelty@{topk}'] = novelty_dict['expl']
    wandb.log(wandb_dict)
    with open(config['eval_folder']+"novelty_distribution.pkl", 'wb') as f:
        pickle.dump(save_dict, f)
    return log_novelty


# def check_update_model(val_acc_dict, test_acc_dict, val_best_acc_dict, test_best_acc_dict, test_expl_pred_dict):
#     val_metric = (val_acc_dict['recall5'] + val_acc_dict['recall10'])/2
#     val_best_metric = (val_best_acc_dict['recall5'] + val_best_acc_dict['recall10'])/2

#     if val_metric > val_best_metric:
#         print('====find better model based on val!====') 
#         val_best_acc_dict = val_acc_dict
#         test_best_acc_dict = test_acc_dict
    
#         # # save prediction eval model
#         # predict_dict = {usr: rec_list for usr, rec_list in zip(test_user_list, test_rec_ranks_list)}
#         with open(config['prediction_folder']+"best_overall_prediction.json", 'w') as f:
#             json.dump(test_expl_pred_dict, f)
#         # acc_dict = {'best_epoch': epoch, 'overall': test_acc_dict}
#         # with open(config['eval_folder']+"best_overall_acc.json", 'w') as f:
#         #     json.dump(acc_dict, f)
#         # torch.save(model.state_dict(), config['model_folder']+"best_overall_acc.pkl")
    
#     metric_list = ['recall5', 'recall10', 'recall20', 'ndcg5', 'ndcg10', 'ndcg20']
#     wandb_dict = dict()
#     for metric in metric_list:
#         wandb_dict[f'best_val_{metric}'] = val_best_acc_dict[metric]
#         wandb_dict[f'best_test_{metric}'] = test_best_acc_dict[metric]
#     wandb.log(wandb_dict)
#     return val_best_acc_dict, test_best_acc_dict

def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')

def create_model(config):
    if config['method'] == 'gru4rec':
        model  = GRU4Rec(config)
    elif config['method'] == 'sasrec':
        model = SASRec(config)
    elif config['method'] == 'bert4rec':
        model = BERT4Rec(config)
    elif config['method'] == 'srgnn':
        model = SRGNN(config)
    elif config['method'] == 'repeatnet':
        model = RepeatNet(config)
    elif config['method'] == 'caser':
        model = Caser(config)
    # elif config['method'] == 'resasrec':
    #     model = RESASRec(config)
    return model

def train_model(model, optimizer, train_data_loader, val_data_loader, test_data_loader, config):
    print(model)
    print(optimizer)

    model = convert_to_gpu(model, device=config['device'])
    model.train()
    start_time = datetime.datetime.now()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    topk_list = [1, 3, 5, 10]
    val_best_acc_dict = {acc+str(topk): 0 for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}
    val_metric_anchor = 'ndcg5'

    # test_best_acc_dict = {acc+str(topk): 0 for acc in ['recall', 'ndcg', 'mrr'] for topk in topk_list}

    test_acc_all_dict = dict()
    # start optimzing 
    for epoch in range(config['epochs']):
        # tqdm_train_loader = tqdm(train_data_loader)
        model.train()
        total_loss = 0
        for step, (user, item_seq, item_seq_len, pos_items, neg_items) in enumerate(train_data_loader):
            item_seq, item_seq_len, pos_items, neg_items  = convert_all_data_to_gpu(item_seq, item_seq_len, pos_items, neg_items, device=config['device'])
            optimizer.zero_grad()
            # print(item_seq)
            # print(item_seq_len)
            loss = model.calculate_loss(item_seq, item_seq_len, pos_items, neg_items)
            total_loss += loss.cpu().data.numpy()
            check_nan(loss)
            loss.backward()
            optimizer.step()
            # print(step)
            # tqdm_train_loader.set_description(f'epoch: {epoch}, train loss: {total_loss / (step + 1)}', refresh=False)

        epoch_avg_loss = total_loss/(step+1)
        scheduler.step(epoch_avg_loss)

        # val eval
        # tqdm_val_loader = tqdm(val_data_loader)
        model.eval() # eval every epoch, can be changed to 100 batches.
        val_rec_ranks_list = []
        val_user_list = []
        val_pos_item_list = []
        val_item_seq_list = []
        for step, (user, item_seq, item_seq_len, pos_items, neg_items) in enumerate(val_data_loader):
            item_seq, item_seq_len, pos_items, neg_items  = convert_all_data_to_gpu(item_seq, item_seq_len, pos_items, neg_items, device=config['device'])
            predict = model.full_sort_predict(item_seq, item_seq_len) # [b, n_items]
            predict = predict.cpu()
            _, rec_ranks = torch.topk(predict, 50, dim=-1) #save top 50
            val_rec_ranks_list.extend(rec_ranks.tolist())
            val_pos_item_list.extend(pos_items.cpu().data.tolist())
            val_user_list.extend(user.data.tolist())
            val_item_seq_list.extend(item_seq.data.tolist())

        val_acc_dict, val_rep_acc_dict, val_expl_acc_dict = eval_acc(topk_list, val_user_list, val_item_seq_list, val_rec_ranks_list, val_pos_item_list)
        
        # test eval
        # tqdm_test_loader = tqdm(test_data_loader)
        model.eval() # eval every epoch, can be changed to 100 batches.
        test_rec_ranks_list = []
        test_user_list = []
        test_pos_item_list = []
        test_item_seq_list = []
        for step, (user, item_seq, item_seq_len, pos_items, neg_items) in enumerate(test_data_loader):
            item_seq, item_seq_len, pos_items, neg_items  = convert_all_data_to_gpu(item_seq, item_seq_len, pos_items, neg_items, device=config['device'])
            predict = model.full_sort_predict(item_seq, item_seq_len) # [b, n_items]
            predict = predict.cpu()
            _, rec_ranks = torch.topk(predict, 50, dim=-1) #save top 50

            test_rec_ranks_list.extend(rec_ranks.tolist())
            test_pos_item_list.extend(pos_items.cpu().data.tolist())
            test_user_list.extend(user.data.tolist())
            test_item_seq_list.extend(item_seq.data.tolist())
            # print (test_rec_ranks_list, test_pos_item_list)
            # raise ValueError
        test_acc_dict, test_rep_acc_dict, test_expl_acc_dict = eval_acc(topk_list, test_user_list, test_item_seq_list, test_rec_ranks_list, test_pos_item_list)

        wandb.log({'epoch': epoch})
        # wandb_dict = {}
        # for topk in [1, 3, 5, 10]:
        #     wandb_dict[f'recall{topk}'] = test_acc_dict[f'recall{topk}']
        #     wandb_dict[f'recall{topk}_rep'] = test_rep_acc_dict[f'recall{topk}']
        #     wandb_dict[f'recall{topk}_expl'] = test_expl_acc_dict[f'recall{topk}']
        # wandb.log(wandb_dict, step=epoch)

        # print(f"Recall1: {test_acc_dict['recall1']} \t Recall1@Rep: {test_rep_acc_dict['recall1']} \t Recall1@Expl: {test_expl_acc_dict['recall1']}")
        # print(f"Recall5: {test_acc_dict['recall5']} \t Recall5@Rep: {test_rep_acc_dict['recall5']} \t Recall5@Expl: {test_expl_acc_dict['recall5']}")
        # print(f"Recall10: {test_acc_dict['recall10']} \t Recall10@Rep: {test_rep_acc_dict['recall10']} \t Recall10@Expl: {test_expl_acc_dict['recall10']}")
        # test_acc_all_dict[epoch] = {'overall': test_acc_dict, 'repeat': test_rep_acc_dict, 'explore': test_expl_acc_dict}
        # find best model epoch, save prediction, test performance, and the model
        if val_acc_dict[val_metric_anchor] > val_best_acc_dict[val_metric_anchor]: 
            val_best_acc_dict = val_acc_dict
            final_predict_dict = {usr: rec_list for usr, rec_list in zip(test_user_list, test_rec_ranks_list)}
            final_user_history_dict = {usr: hist_seq for usr, hist_seq in zip(test_user_list, test_item_seq_list)}
            final_user_pos_dict = {usr:pos_item for usr, pos_item in zip(test_user_list, test_pos_item_list)}
            # # save prediction
            with open(config['prediction_folder']+"best_overall_prediction.pkl", 'wb') as f:
                pickle.dump(final_predict_dict, f)
            # # save performance.
            best_acc_dict = {'best_epoch': epoch, 'overall': test_acc_dict, 'repeat': test_rep_acc_dict, 'explore': test_expl_acc_dict}
            with open(config['eval_folder']+"best_overall_acc.json", 'w') as f:
                json.dump(best_acc_dict, f)
            # # save model pkl
            # torch.save(model.state_dict(), config['model_folder']+"best_overall_acc.pkl")
    wandb_dict = {}
    for metric in ['recall', 'ndcg', 'mrr']:
        for topk in [1, 3, 5, 10]:
            wandb_dict[f'best_{metric}{topk}'] = best_acc_dict['overall'][f'{metric}{topk}']
            wandb_dict[f'best_{metric}{topk}_rep'] = best_acc_dict['repeat'][f'{metric}{topk}']
            wandb_dict[f'best_{metric}{topk}_expl'] = best_acc_dict['explore'][f'{metric}{topk}']
    wandb.log(wandb_dict)
    print('finish training! evaluation started!')
    ## start compute user-centered novelty and item-centered fresh exposure!
    item_label_dict, golden_fresh_ratio_dict = label_merit_distribution(final_user_history_dict, final_user_pos_dict)
    log_novelty = eval_novelty(final_predict_dict, final_user_history_dict, final_user_pos_dict, topk_list)
    toplabelk_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    log_exposure = eval_exposure(final_predict_dict, final_user_history_dict, final_user_pos_dict, item_label_dict, golden_fresh_ratio_dict, topk_list, toplabelk_list, config)
    # with open(config['eval_folder']+"test_acc_every_epoch", 'w') as f:
    #     json.dump(test_acc_all_dict, f)
    end_time = datetime.datetime.now()
    print(f'End! Total cost {end_time-start_time}!')    
                    
if __name__ == '__main__':

    config = get_config()
    wandb.init(project="fresh-exposure", 
                name=f"{config['method']}_{config['dataset']}_{config['loss_type']}_mode{config['train_mode']}_shared{config['shared_emb']}_fold{config['foldk']}",
                config=config)
    config = wandb.config
    model = create_model(config)
    os.environ['CUDA_LAUNCH_BLOCKING']= '1'

    train_data_loader, val_data_loader, test_data_loader = get_dataloader(config, d_type='train'), get_dataloader(config, d_type='val'), get_dataloader(config, d_type='test')
    print('dataloader generated!')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    train_model(model, optimizer, train_data_loader, val_data_loader, test_data_loader, config)
    



    
