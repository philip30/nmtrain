import chainer
import chainer.functions as F

import nmtrain

class LSTMDecoder(chainer.Chain):
  def __init__(self, out_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(LSTMDecoder, self).__init__(
      decoder       = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
      affine_vocab  = chainer.links.Linear(hidden_size, out_size),
      output_embed  = chainer.links.EmbedID(out_size, embed_size)
    )

  def __call__(self):
    y = F.softmax(self.affine_vocab(F.tanh(self.h)))
    return Output(y=y)

  def init(self, h):
    self.decoder.reset_state()
    self.h = self.decoder(h)

  def update(self, next_word):
    self.h = self.decoder(self.output_embed(next_word))

# Implementation of Luong et al.
class LSTMAttentionalDecoder(LSTMDecoder):
  def __init__(self, out_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(LSTMDecoder, self).__init__(
      decoder         = nmtrain.chner.StackLSTM(embed_size + hidden_size, hidden_size, lstm_depth, dropout_ratio),
      context_project = chainer.links.Linear(2*hidden_size, hidden_size),
      affine_vocab    = chainer.links.Linear(hidden_size, out_size),
      output_embed    = chainer.links.EmbedID(out_size, embed_size)
    )

  def init(self, h):
    h, S = h
    self.decoder.reset_state()
    self.S = S
    self.h = self.decoder(h)

  def __call__(self):
    # Calculate Attention vector
    a = F.squeeze(F.softmax(F.batch_matmul(self.S, self.h)), axis=2)
    # Calculate context vector
    c = F.squeeze(F.batch_matmul(self.S, a, transa=True), axis=2)
    # Calculate hidden vector + context
    self.ht = self.context_project(F.concat((self.h, c), axis=1))
    # Calculate Word probability distribution
    y = F.softmax(self.affine_vocab(F.tanh(self.ht)))
    # Return the vocabulary size output projection
    return Output(y=y, a=a)

  def update(self, next_word):
    # embed_size + hidden size -> input feeding approach
    self.h = self.decoder(F.hstack((self.output_embed(next_word), self.ht)))

# MISC class for holding the output
class Output(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

