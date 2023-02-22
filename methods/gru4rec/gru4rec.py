import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from recbole.model.loss import BPRLoss

class GRU4Rec(nn.Module):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.
    Note:
        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.loss_type = config['loss_type']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']

        # dataset info
        self.n_items = config['n_items']
        self.max_seq_length = config['max_seq_length']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        
        self.shared_emb = config['shared_emb']
        if config['shared_emb'] == 0:
            self.last_layer_item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, item_seq, item_seq_len, pos_items, neg_items=None):
        seq_output = self.forward(item_seq, item_seq_len)
        if self.loss_type == 'BPR':
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            if self.shared_emb == 1:
                test_item_emb = self.item_embedding.weight # [item_num H]
            else:
                test_item_emb = self.last_layer_item_embedding.weight  # [item_num H]

            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # B*K B
            loss = self.loss_fct(logits, pos_items)
            return loss

    # def predict(self, item_seq, item_seq_len, test_item):
    #     seq_output = self.forward(item_seq, item_seq_len)
    #     test_item_emb = self.item_embedding(test_item)
    #     scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
    #     return scores

    def full_sort_predict(self, item_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        if self.shared_emb == 1:
            test_item_emb = self.item_embedding.weight  # [item_num H]
        else:
            test_item_emb = self.last_layer_item_embedding.weight # [item_num H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, n_items]
        return scores
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask