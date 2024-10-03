import numpy as np
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
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

    def make_subsequent_mask(self, query, key):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)

        return mask

    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(query=src, key=src)
        return pad_mask

    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(query=tgt, key=tgt)
        seq_mask = self.make_subsequent_mask(query=tgt, key=tgt)
        mask = pad_mask & seq_mask
        return mask

    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(query=tgt, key=src)
        return pad_mask
        
    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return out

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)
        return out

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        out = self.generator(decoder_out)
        # out = F.log_softmax(out, dim=-1)
        return out, decoder_out