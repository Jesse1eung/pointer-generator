# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import os
import time
import argparse
import tensorflow as tf

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from models.model import Model
from utils import config
from utils.dataset import Vocab
from utils.dataset import Batcher
from utils.utils import get_input_from_batch
from utils.utils import get_output_from_batch
from utils.utils import calc_running_avg_loss

use_cuda = config.use_gpu and torch.cuda.is_available()
gpus = [0, 1, 2, 3]

torch.cuda.set_device('cuda:{}'.format(gpus[0]))

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(self.vocab, config.train_data_path,
                               config.batch_size, single_pass=True, mode='train')
        time.sleep(10)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'models')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoders_state_dict': self.model.encoders.state_dict(),
            'encoders_att_state_dict': self.model.encoders_att.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_path=None):
        self.model = Model(model_path, is_tran= config.tran)
        initial_lr = config.lr_coverage if config.is_coverage else config.lr

        params = list(self.model.encoders.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        total_params = sum([param[0].nelement() for param in params])
        print('The Number of params of model: %.3f million' % (total_params / 1e6))  # million
        self.optimizer = optim.Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_lens, enc_pos, enc_padding_mask, enc_batch_extend_vocab, \
        extra_zeros, c_t, coverage = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_lens, dec_pos, dec_padding_mask, max_dec_len, tgt_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        enc_outs = []
        enc_feas = []
        enc_hs = []
        if not config.tran:
            for i, encoder in enumerate(self.model.encoders):
                enc_out, enc_fea, enc_h = encoder(enc_batch[i], enc_lens[i])
                enc_outs.append(enc_out)
                enc_feas.append(enc_fea)
                enc_hs.append(enc_h)

        # else:
        #     enc_out, enc_fea, enc_h = self.model.encoder(enc_batch, enc_pos)



        s_t = self.model.reduce_state(enc_h)

        step_losses, cove_losses = [], []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t = dec_batch[:, di]  # Teacher forcing

            # modify the original frame for two encoders.
            final_dist_0, s_t_0, c_t_0, attn_dist_0, p_gen, next_coverage = \
                self.model.decoder(y_t, s_t, enc_outs[0], enc_feas[0], enc_padding_mask[0], c_t,
                                   extra_zeros, enc_batch_extend_vocab[0], coverage, di)

            final_dist_1, s_t_1, c_t_1, attn_dist, p_gen, next_coverage = \
                self.model.decoder(y_t, s_t, enc_outs[1], enc_feas[1], enc_padding_mask[1], c_t,
                                   extra_zeros, enc_batch_extend_vocab[1], coverage, di)

            y_t_emb = self.model.decoder.tgt_word_emb(y_t)
            encoders_att = self.model.encoders_att(enc_hs, y_t_emb)
            final_dist = torch.stack((final_dist_0, final_dist_1), dim=1)
            final_dist = torch.bmm(encoders_att, final_dist).squeeze()

            c_t = torch.stack((c_t_0, c_t_1), 1)
            c_t = torch.bmm(encoders_att, c_t).squeeze()
            encoders_att_ = encoders_att.transpose(0, 1).contiguous() # 1 x b x 2
            h = s_t_0[0] * encoders_att_[:, :, :1] + s_t_1[0] * encoders_att_[:, :, 1:]
            c = s_t_0[1] * encoders_att_[:, :, :1] + s_t_1[1] * encoders_att_[:, :, 1:]
            s_t = (h, c)

            tgt = tgt_batch[:, di]
            step_mask = dec_padding_mask[:, di]
            gold_probs = torch.gather(final_dist, 1, tgt.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                cove_losses.append(step_coverage_loss * step_mask)
                coverage = next_coverage

            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        clip_grad_norm_(self.model.encoders.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        if config.is_coverage:
            cove_losses = torch.sum(torch.stack(cove_losses, 1), 1)
            batch_cove_loss = cove_losses / dec_lens
            batch_cove_loss = torch.mean(batch_cove_loss)
            return loss.item(), batch_cove_loss.item()

        return loss.item(), 0.

    def run(self, n_iters, model_path=None):
        iter, running_avg_loss = self.setup_train(model_path)
        start = time.time()
        interval = 500

        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss, cove_loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % interval == 0:
                self.summary_writer.flush()
                print(
                    'step: %d, second: %.2f , loss: %f, cover_loss: %f' % (iter, time.time() - start, loss, cove_loss))
                start = time.time()
            if iter % 500 == 0:
                self.save_model(running_avg_loss, iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_path",
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()

    train_processor = Train()
    train_processor.run(config.max_iterations, args.model_path)
