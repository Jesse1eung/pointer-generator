# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from utils import config
from numpy import random
from models.layers import Encoder
from models.layers import Encoders_Attention
from models.layers import Decoder
from models.layers import ReduceState

from transformer.model import TranEncoder

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class Model(object):
    def __init__(self, model_path=None, is_eval=False, is_tran = False):
        # encoder = Encoder()
        encoders = nn.ModuleList([Encoder() for _ in range(config.num_encoders)])
        encoders_att = Encoders_Attention()
        decoder = Decoder()
        reduce_state = ReduceState()
        if is_tran:
            encoder = TranEncoder(config.vocab_size, config.max_enc_steps, config.emb_dim,
                 config.n_layers, config.n_head, config.d_k, config.d_v, config.d_model, config.d_inner)

        # shared the embedding between encoders and decoder
        for i in range(1, config.num_encoders):
            encoders[i].src_word_emb.weight = encoders[0].src_word_emb.weight
        decoder.tgt_word_emb.weight = encoders[0].src_word_emb.weight

        if is_eval:
            encoders = encoders.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoders = encoders.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoders = encoders
        self.decoder = decoder
        self.encoders_att = encoders_att
        self.reduce_state = reduce_state

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            self.encoders.load_state_dict(state['encoders_state_dict'])
            self.encoders_att.load_state_dict(state['encoders_att_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
