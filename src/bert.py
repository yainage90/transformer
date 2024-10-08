import torch.nn as nn
from base_module import *

class Bert(nn.Module):
    def __init__(self, embed, encoder, generator):
        super(Bert, self).__init__()
        self.embed = embed
        self.encoder = encoder
        self.generator = generator

    def make_pad_mask(self, query, key, pad_idx=0):
        # query, key는 token sequence(정수)로 입력됨.
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len) # (n_batch, 1, query_seq_len, key_seq_len)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

        mask = query_mask * key_mask
        mask.requires_grad = False

        return mask

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(query=src, key=src)
        return pad_mask

    def encode(self, src, src_mask):
        out = self.encoder(self.embed(src), src_mask)
        return out

    def forward(self, src):
        src_mask = self.make_src_mask(src)
        out = self.encode(src, src_mask)
        out = self.generator(out)
        return out