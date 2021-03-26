# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from utils import config
from models.basic import BasicModule
from models.attention import MultiAttention


class Encoder(BasicModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        self.init_params()

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.src_word_emb(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x l x n
        encoder_outputs = encoder_outputs.contiguous()

        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B*l x 2*hidden_dim
        encoder_feature = self.fc(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class Encoders(BasicModule):
    def __init__(self):
        super(Encoders, self).__init__()

        self.encoders = nn.ModuleList([Encoder() for _ in range(config.num_encoders)])

    def forward(self, batch, seq_lens):
        enc_outs = []
        enc_feas = []
        enc_hs = []
        for i, encoder in enumerate(self.encoders):
            enc_out, enc_fea, enc_h = encoder(batch[i], seq_lens[i])
            enc_outs.append(enc_out)
            enc_feas.append(enc_fea)
            enc_hs.append(enc_h)

        return enc_outs, enc_feas, enc_hs


class Encoders_Attention(BasicModule):
    def __init__(self):
        super(Encoders_Attention, self).__init__()

        self.fc = nn.Linear(config.hidden_dim * 2, 1)
        self.w_y = nn.Linear(config.emb_dim, config.hidden_dim * 2)
        self.w_h = nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2)

        # self.
        self.init_params()

    def forward(self, enc_hs, y_t_embed):
        y_fea = self.w_y(y_t_embed)  # b x 2*hidden
        h_feas = []
        for enc_h in enc_hs:
            h, c = enc_h  # 2 x b x hidden
            h = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)  # b x 2*hidden
            c = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
            h_fea = torch.cat((h, c), dim=1).unsqueeze(1)  # b  x 1 x 4*hidden
            h_feas.append(h_fea)
        h_feas = torch.cat(h_feas, dim=1)  # b x 2 x 4*hidden
        h_feas = self.w_h(h_feas)  # b x 2 x 2*hidden

        y_fea_expanded = y_fea.unsqueeze(1).expand_as(h_feas).contiguous()
        att_fea = h_feas + y_fea_expanded
        e = torch.tanh(att_fea)
        scores = self.fc(e).transpose(1, 2).contiguous()  # b x 1 x 2
        att_dist = F.softmax(scores, dim=-1)

        return att_dist


class ReduceState(BasicModule):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.init_params()

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)  # h, c dim = 1 x b x hidden_dim


class Decoder(BasicModule):
    def __init__(self):
        super(Decoder, self).__init__()

        self.multiattention = MultiAttention()
        self.encoders_att = Encoders_Attention()
        # decoder
        self.tgt_word_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.con_fc = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, batch_first=True, bidirectional=False)

        if config.pointer_gen:
            self.p_gen_fc = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.fc1 = nn.ModuleList(
            [nn.Linear(config.hidden_dim * 3, config.hidden_dim) for _ in range(config.num_encoders)])
        self.fc2 = nn.ModuleList([nn.Linear(config.hidden_dim, config.vocab_size) for _ in range(config.num_encoders)])

        self.init_params()

    def forward(self, y_t, s_t, enc_out_tuple, enc_padding_mask,
                c_t, extra_zeros, enc_batch_extend_vocab, coverage, step):
        (enc_out, enc_fea, enc_h) = enc_out_tuple
        if not self.training and step == 0:
            dec_h, dec_c = s_t
            s_t_hat = torch.cat((dec_h.view(-1, config.hidden_dim),
                                 dec_c.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            # c_t, _, coverage_next = self.attention_network(s_t_hat, enc_out, enc_fea,
            #                                                enc_padding_mask, coverage)
            print("step 0")
            # coverage = coverage_next

        y_t_embd = self.tgt_word_emb(y_t)

        x = self.con_fc(torch.cat((c_t, y_t_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t)

        dec_h, dec_c = s_t
        s_t_hat = torch.cat((dec_h.view(-1, config.hidden_dim),
                             dec_c.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        encs_att = self.encoders_att(enc_h, y_t_embd)
        c_t, c_t_list, attn_dist, coverage_next = self.multiattention(s_t_hat, enc_out, enc_fea, encs_att,
                                                            enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None

        p_gen, final_dist = self.merge_final_dist(c_t_list, s_t_hat, x, attn_dist, encs_att, extra_zeros,
                                           enc_batch_extend_vocab, lstm_out, p_gen)

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

    def merge_final_dist(self, c_t, s_t_hat, x, attn_dist, encs_att, extra_zeros, enc_batch_extend_vocab,
                         lstm_out, p_gen=None):
        final_dists = []

        for i in range(config.num_encoders):
            if config.pointer_gen:
                p_gen_inp = torch.cat((c_t[i], s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
                p_gen = self.p_gen_fc(p_gen_inp)
                p_gen = torch.sigmoid(p_gen)
            output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t[i]), 1)  # B x hidden_dim * 3
            output = self.fc1[i](output)  # B x hidden_dim

            output = self.fc2[i](output)  # B x vocab_size
            vocab_dist = F.softmax(output, dim=1)

            if config.pointer_gen:
                vocab_dist_ = p_gen * vocab_dist
                attn_dist_ = (1 - p_gen) * attn_dist[i]

                if extra_zeros is not None:
                    vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

                final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab[i], attn_dist_)
            else:
                final_dist = vocab_dist
            final_dists.append(final_dist)

        final_dists = torch.stack(final_dists, dim=1)
        final_dists_reweighted = torch.bmm(encs_att, final_dists).squeeze()

        return p_gen, final_dists_reweighted
