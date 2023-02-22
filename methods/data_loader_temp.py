from audioop import reverse
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch import nn

# this data loader split the dataset according to leave one out strategy.
class SeqDataset(Dataset):

    def __init__(self, config, type=None):

        super().__init__()
        self.data_path = config['data_path']
        # self.foldk_path = config['foldk_path'] # use this instead of random seed for better analysis
        self.foldk = config['foldk']
        self.max_seq_length = config['max_seq_length']
        # self.filter_repeat = config['filter_repeat'] # for train, this is to manuplate the dataset its own, not to attack the dataset.
        self.train_mode = config['train_mode']

        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        # with open(self.foldk_path, 'rb') as f:
        #     foldk_data = pickle.load(f)
        
        # users = foldk_data[self.foldk][type]
        self.user_list = []
        self.item_seq_list = []
        self.item_seq_length_list = []
        self.pos_item_list = []

        for usr_id, seq in data.items():
            if type == 'train':
                item_seq = data[usr_id][:-3]
                pos_item = data[usr_id][-3]
            elif type == 'val':
                item_seq = data[usr_id][:-2]
                pos_item = data[usr_id][-2]
            elif type == 'test':
                item_seq = data[usr_id][:-1]
                pos_item = data[usr_id][-1]

            if (pos_item in item_seq) and (self.train_mode == 1):
                continue # drop repeat samples during training only keep new items.
            self.user_list.append(usr_id)
            self.item_seq_list.append(torch.tensor(item_seq))
            self.item_seq_length_list.append(len(item_seq))
            self.pos_item_list.append(pos_item)

    def __getitem__(self, index):
        user = self.user_list[index]
        item_seq = self.item_seq_list[index]
        item_seq_length = self.item_seq_length_list[index]
        pos_item = self.pos_item_list[index]
        neg_item = 0 # placeholder, do not use bpr now
        return user, item_seq, item_seq_length, pos_item, neg_item
    
    def __len__(self):
        return len(self.item_seq_list)

def collate_fn(batch_data):
    ret = list()
    for idx, data_list in enumerate(zip(*batch_data)):
        if isinstance(data_list[0], torch.Tensor) and idx==1:
            # data_list.sort(key=lambda data: len(data), reverse=True)
            item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            # print(item_seq)
            ret.append(item_seq)
        else:
            ret.append(torch.tensor(data_list))
    # (user, item_seq, item_seq_length, pos_item, neg_item) --> tensor format
    return tuple(ret) 

def collate_fn_max_len50(batch_data):
    ret = list()
    for idx, data_list in enumerate(zip(*batch_data)):
        if isinstance(data_list[0], torch.Tensor) and idx==1:
            # print(data_list[0])
            # print(nn.ConstantPad1d((0, 50-data_list[0].shape[0]), 0)(data_list[0]))
            data_list = list(data_list)
            data_list[0] = nn.ConstantPad1d((0, 50-data_list[0].shape[0]), 0)(data_list[0])
            data_list = tuple(data_list)
            # data_list.sort(key=lambda data: len(data), reverse=True)
            item_seq = rnn_utils.pad_sequence(data_list, batch_first=True, padding_value=0)
            # print(item_seq)
            ret.append(item_seq)
        else:
            ret.append(torch.tensor(data_list))
    # (user, item_seq, item_seq_length, pos_item, neg_item) --> tensor format
    return tuple(ret) 

def get_dataloader(config, d_type=None):
    dataset = SeqDataset(config, type=d_type)
    print(f'{d_type} data length -> {len(dataset)}')
    if (config['method'] == 'repeatnet') or (config['method'] == 'caser'):
        data_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False, collate_fn=collate_fn_max_len50)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False, collate_fn=collate_fn)
    return data_loader