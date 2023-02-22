from datetime import datetime
import pandas as pd
import argparse
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='diginetica', help='name of the dataset')
    parser.add_argument('--min_item_num', type=int, default=5)
    parser.add_argument('--min_seq_len', type=int, default=5)
    parser.add_argument('--max_seq_len', type=int, default=50)
    args = parser.parse_args()
    dataset, min_item_num, min_seq_len, max_seq_len = args.dataset, args.min_item_num, args.min_seq_len, args.max_seq_len

    data_path  = f'{dataset}/{dataset}-all.csv'
    all_data = pd.read_csv(data_path, index_col=False)
    all_data = all_data[['user_id', 'item_id', 'timestamp']]
    # print(all_data.head(10))
    # filter items by count    
    # sort and group
    all_data_sorted = all_data.sort_values(['user_id', 'timestamp'], ignore_index=True)
    print(all_data_sorted.head(30))

    # create uid - item_seq
    data_list = all_data_sorted.values.tolist()
    
    old_uid = data_list[0][0]
    u_i_seq_dict_all = dict()
    i_seq = []

    test_iid_set = set()
    for (uid, iid, timestamp) in tqdm(data_list):
        if uid != old_uid:
            u_i_seq_dict_all[old_uid] = i_seq
            old_uid = uid
            i_seq = []
            i_seq.append(iid)
            # if iid in item_remap_dict.keys():
            #     i_seq.append(iid)
            test_iid_set.add(last_iid)
        else:
            i_seq.append(iid)
            last_iid = iid
            # if iid in item_remap_dict.keys():
            #     i_seq.append(iid)
    
    # frist remove users whose purchase item is below 5
    ui_seq_dict_filter_user = dict()
    item_cnt_dict = dict()
    for uid, i_seq in u_i_seq_dict_all.items():
        if len(i_seq)>=min_seq_len:
            ui_seq_dict_filter_user[uid] = i_seq
            for item in i_seq:
                if item not in item_cnt_dict.keys():
                    item_cnt_dict[item] = 1
                else:
                    item_cnt_dict[item] += 1

    # get item set above minmum threshold
    items_above_min = set()
    for item, cnt in item_cnt_dict.items():
        if cnt>= min_item_num:
            items_above_min.add(item)
    

    # remap itemid start from 1, 0 is for the empty holder
    item_remap_dict = dict()
    item_ind = 1
    for item in items_above_min:
        item_remap_dict[item] = item_ind
        item_ind += 1

    # filter data by seq length, and reindex user id.
    user_ind = 1
    user_remap_dict = dict()
    u_i_seq_dict_filtered = dict()
    for uid, i_seq in ui_seq_dict_filter_user.items():
        new_i_seq = []
        # remap item_seq
        for item in i_seq:
            if item in item_remap_dict.keys():
                new_i_seq.append(item_remap_dict[item])

        if len(new_i_seq) >= min_seq_len:
            if len(new_i_seq) > max_seq_len:
                new_i_seq = new_i_seq[-max_seq_len:].copy()
            u_i_seq_dict_filtered[user_ind] = new_i_seq
            user_remap_dict[uid] = user_ind
            user_ind += 1
    
    # save to pkl format
    remap_dict = {'user': user_remap_dict, 'item': item_remap_dict}
    with open(f'{dataset}/ui_remap.pkl', 'wb') as f:
        pickle.dump(remap_dict, f)
    with open(f'{dataset}/{dataset}-filter-new.pkl', 'wb') as f:
        pickle.dump(u_i_seq_dict_filtered, f)

