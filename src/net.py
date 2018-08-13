import collections
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import training
from chainer import Variable

tokens = collections.Counter({
    '<UNK>': 0,
    '<SOS>': 1,
    '<EOS>': 2
})


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_selection = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_selection, 0)

    return exs


class Encoder (chainer.Chain):
    def __init__(self, n_vocab, n_layers, n_hidden, dropout_ratio):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab, n_hidden, ignore_label=-1)
            self.bilstm = L.NStepBiLSTM(n_layers, n_hidden, n_hidden, dropout_ratio)

    def __call__(self, xs):
        batch_size, max_length = xs.shape

        exs = self.embed_x(xs)

class AttentionModule(chainer.Chain):
    def __init__(self, n_hidden):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.fbh = L.Linear(n_hidden, n_hidden*2)
            self.hh = L.Linear(n_hidden, n_hidden*2)
            self.hw = L.Linear(n_hidden*2, 1)

        self.n_hidden = n_hidden

    def __call__(self, enc_h, dec_h):
        batch_size = len(dec_h)

        ws = []
        sum_w = Variable(self.xp.zeros((batch_size, 1), dtype='float32'))

        for i, h in enumerate(enc_h):
            pad_num = batch_size - h.shape[0]
            enc_h[i] = F.pad(h, [(0, pad_num), (0, 0)], 'constant')

        for h in enc_h:
            w = F.tanh(self.fbh(h) + self.hh(dec_h))
            w = F.exp(self.hw(w))
            ws.append(w)
            sum_w += w

        att = Variable(self.xp.zeros((batch_size, self.n_hidden), dtype='float32'))

        for fb, w in zip(enc_h, ws):
            w /= sum_w
            att += F.reshape(F.batch_matmul(fb, w), (batch_size, self.n_hidden))

        return att


class seq2seq(chainer.Chain):
    def __init__(
            self,
            n_layers,
            n_source_vocab,
            n_target_vocab,
            n_hidden,
            dropout_ratio=0.2,
    ):

        super(seq2seq, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_ratio = dropout_ratio

        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_hidden, ignore_label=-1)
            self.embed_y = L.EmbedID(n_target_vocab, n_hidden*2, ignore_label=-1)
            self.fh_encoder = L.LSTM(n_hidden, n_hidden)
            self.bh_encoder = L.LSTM(n_hidden, n_hidden)
            self.decoder = L.LSTM(n_hidden*4, n_hidden*2)
            self.W = L.Linear(n_hidden*2, n_target_vocab)
            self.attention = AttentionModule(n_hidden*2)

    def __call__(self, xs, ys):
        batch = len(xs)
        # delete <SOS> and <EOS> from x and reverse the order

        xsf = [x[1:-1] for x in xs]
        xsb = [x[1:-1][::-1] for x in xs]

        ys_in = [y[:-1] for y in ys]
        ys_out = [y[1:] for y in ys]

        exs_f = sequence_embed(self.embed_x, xsf)
        exs_b = sequence_embed(self.embed_x, xsb)
        eys = sequence_embed(self.embed_y, ys_in)

        # sort decriasing order for LSTM input to accept
        # sequences of variable length.
        # and transpose sequence to get time t's input batch.
        exs_f = F.transpose_sequence(sorted(exs_f, key=len, reverse=True))
        exs_b = F.transpose_sequence(sorted(exs_b, key=len, reverse=True))
        eys = F.transpose_sequence(sorted(eys, key=len, reverse=True))

        # padding eys for calculating attention.

        f_list = []
        b_list = []

        self.fh_encoder.reset_state()
        for ex in exs_f:
            f = self.fh_encoder(ex)
            f = F.dropout(f, self.dropout_ratio)
            f_list.append(f)
        self.bh_encoder.reset_state()
        for ex in exs_b:
            b = self.bh_encoder(ex)
            b = F.dropout(b, self.dropout_ratio)
            b_list.append(b)

        fb_list = [F.concat([f_list[i], b_list[i]], axis=1) for i in range(len(exs_f))]
        # concat_fb = fb_list
        # concat_fb = F.concat(fb_list, axis=0)

        dec_h = Variable(self.xp.zeros((batch, self.n_hidden*2), dtype='float32'))

        # TODO: how to feed encoder's cell and hidden state into decoder ?
        h_list = []
        for h_w in eys:
            att = self.attention(fb_list, dec_h)

            # padding for calculating attention.
            if att.shape != h_w.shape:
                pad_num = att.shape[0] - h_w.shape[0]
                h_w = F.pad(h_w, [(0, pad_num), (0, 0)], 'constant')

            h_w = F.concat([att, h_w], axis=1)
            dec_h = self.decoder(h_w)
            h_list.append(dec_h)
        print(len(h_list))
        h_list = F.concat(h_list, 0)
        h_list = F.transpose(h_list)

        concat_os = F.concat(h_list, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        print(len(concat_os))
        print(len(concat_ys_out))
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce="no")) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        prep = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'prep': prep}, self)

        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # delete <SOS> and <EOS> from x and reverse the order
            xs = [x[1:-1][::-1] for x in xs]

            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, tokens['<SOS>'], np.int32)

            result = []
            for _ in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, axis=0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.l1(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                result.append(ys)

            result = cuda.to_cpu(self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

            outs = []
            for y in result:
                inds = np.argwhere(y == tokens['<EOS>'])
                if len(inds) > 0:
                    y = y[:inds[0, 0]]
                outs.append(y)

            return outs
