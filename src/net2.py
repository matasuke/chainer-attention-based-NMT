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


class Seq2Seeq(chainer.Chain):

    def __init__(
            self,
            n_layers,
            n_source_vocab,
            n_target_vocab,
            n_encoder_units,
            dropout,
            n_decoder_units,
            n_attention_units,
            n_maxout_units
    ):
        super(Seq2Seq, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(
                n_layer,
                n_source_vocab,
                n_encoder_units,
                dropout
            )

            self.decoder = Decoder(
                n_target_vocab,
                n_decoder_units,
                n_attention_units,
                n_encoder_units*2,
                n_maxout_units
            )

    def __call__(self, xs, ys):
        batch_size = len(xs)

        xs = [x[1:-1] for x in xs]

        ys_in = [y[:-1] for y in ys]
        ys_out = [y[1:] for y in ys]

        hxs = self.encoder(xs)
        os = self.decoder(ys_in, hxs)

        concat_os = F.concat(os, axis=0)
        concat_ys = F.flattend(ys.T)
        n_words = len(self.xp.where(concat_ys.data != -1)[0])

class Encoder(chainer.Chain):

    def __init__(self, n_layers, n_vocab, n_units, dropout):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedId(n_vocab, u_units, ignore_label=-1)
            self.bilstm = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, xs):
        batch_size, max_length = xs.shape

        exs = self.embed_x(xs)
        exs = F.separate(exs, axis=0)
        masks = self.xp.vsplit(xs != -1, batch_size)
        masked_exs = [ex[mask.reshape((-1, ))] for ex, mask in zip(exs, masks)]

        _, _, hxs = self.bilstm(None, None, masked_exs)
        hxs = F.pad_sequences(hxs, length=max_length, padding=0.0)

        return hxs


class Decoder(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_attention_units,
                 n_encoder_output_units, n_maxout_units, n_maxout_pools=2):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed_y = L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.lstm = L.StatelessLSTM(n_units+n_encoder_output_units, n_units)
            self.maxout = L.Maxout(n_units+n_encoder_output_units+u_units,
                                   n_maxout_units, n_maxout_pools)
            self.w = L.Linear(n_maxout_units, n_vocab)
            self.attention = AttentionModule(n_encoder_output_units, n_attention_units, n_units)
        self.n_units = n_units

    def __call__(self, ys, hxs):
        batch_size, max_length, encoder_output_size = hxs.shape
        compute_context = self.attention(hxs)
        c = Varialbe(self.xp.zeros((batch_size, self.n_units), 'f'))
        h = Variable(self.xp.zeros((batch_size, self.n_units)), 'f')

        os = []
        for y in self.xp.hsplit(ys, ys.shape[1]):
            y = y.reshape((batch_size, ))
            eys = self.embed_y(y)
            context = conpute_context(eys)
            concatenated = F.concat([eys, context])

            c, h = self.lstm(c, h, concatenated)
            concatenated = F.concat([concatenated, h])
            o = self.w(self.maxout(concatenated))

        return os

    def translate(self, hsx, max_length=100):
        batch_size, _, _ = hxs.shape
        compute_context = self.attention(hxs)
        c = Varialbe(self.xp.zeros((batch_size, self.n_units), 'f'))
        h = Variable(self.xp.zeros((batch_size, self.n_units)), 'f')

        ys = self.xp.full(batch_size, tokens['<SOS>'], np.int32)

        results = []
        for _ in range(max_length):
            eys = self.embed_y(ys)

            context = conpute_context(h)
            concatenated = F.concat([eys, context])

            c, h = self.lstm(c, h, concatenated)
            concatenated = F.concat([concatenated, h])

            logit = self.w(self.maxout(concatenated))
            y = F.reshape(F.argmax(logit, axis=1), (batch_size, ))

            results.append(y)

        results = F.separate(F.transpose(F.vstack(results)), axis=0)

        outs = []
        for y in results:
            inds = np.argwhere(y == tokens['<EOS>'])
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def AttentionModule(chainer.Chain):

    def __init__(self, n_encoder_output_units,
                 n_attention_units, n_decoder_units):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.h = L.Linear(n_encoder_output_units, n_attention_units)
            self.s = L.Linear(n_decoder_units, n_attention_units)
            self.o = L.Linear(n_attention_units, 1)

        self.n_encoder_output_units = n_encoder_output_units
        self.n_attention_units = n_attention_units

    def __call__(self, hxs):
        batch_size, max_length, encoder_output_size = hxs.shape

        encoder_factor = F.reshape(
            self.h(
                F.reshape(
                    hxs,
                    (batch_size*max_length, self.n_encoder_output_size)
                )
            ),
            (batch_size, max_length, self.n_attention_units)
        )

        def compute_context(dec_hidden):
            decoder_factor = F.broadcast_to(
                F.reshape(
                    self.s(dec_hidden),
                    (batch_size, 1, n_attention_units)
                ),
                (batch_size, max_length, self.n_attention_units)
            )

            attention = F.softmax(
                F.reshape(
                    self.o(
                        F.reshape(
                            F.tanh(encoder_factor + decoder_factor),
                            (batch_size*max_length, self.n_attention_units)
                        )
                    ),
                    (batch_size, max_length)
                )
            )

            context = F.reshape(
                F.batch_matmul(attention, hxs, transa=True),
                (batch_size, encoder_output_size)
            )

            return context

        return compute_context

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_selection = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_selection, 0)

    return exs
