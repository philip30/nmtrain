import chainer
import chainer.functions as F

import nmtrain

class LSTMDecoder(chainer.Chain):
  def __init__(self, out_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(LSTMDecoder, self).__init__(
      decoder       = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
      affine_vocab  = chainer.links.Linear(hidden_size, out_size),
      output_embed  = chainer.links.EmbedID(out_size, embed_size),
      state_init    = chainer.links.Linear(hidden_size, embed_size)
    )

  def __call__(self):
    y = F.softmax(self.affine_vocab(F.tanh(self.h)))
    return Output(y=y)

  def init(self, h):
    self.decoder.reset_state()
    self.h = self.decoder(self.state_init(h))

  def update(self, next_word):
    self.h = self.decoder(self.output_embed(next_word))

# Implementation of Luong et al.
class LSTMAttentionalDecoder(LSTMDecoder):
  def __init__(self, out_size, embed_size, hidden_size,
               dropout_ratio, lstm_depth, input_feeding=True,
               attention_type="dot"):
    decoder_in_size = embed_size
    if input_feeding:
      decoder_in_size += hidden_size
 
    if attention_type == "dot":
      attention = DotAttentionLayer()
    elif attention_type == "general":
      attention = GeneralAttentionLayer(hidden_size)
    elif attention_type == "mlp":
      attention = MLPAttentionLayer(hidden_size)
    else:
      raise ValueError("Unknown Attention Type:", attention_type)

    super(LSTMDecoder, self).__init__(
      decoder         = nmtrain.chner.StackLSTM(decoder_in_size, hidden_size, lstm_depth, dropout_ratio),
      context_project = chainer.links.Linear(2*hidden_size, hidden_size),
      affine_vocab    = chainer.links.Linear(hidden_size, out_size),
      output_embed    = chainer.links.EmbedID(out_size, embed_size),
      attention       = attention
    )
    self.input_feeding = input_feeding
    self.dropout_ratio = dropout_ratio

  def init(self, h):
    h, S = h
    self.decoder.reset_state()
    self.S = S
    self.h = self.decoder(F.dropout(h,
                                    ratio=self.dropout_ratio,
                                    train=nmtrain.environment.is_train()))

  def __call__(self):
    # Calculate Attention vector
    a = self.attention(self.S, self.h)
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
    decoder_update = self.output_embed(next_word)
    if self.input_feeding:
      decoder_update = F.hstack((decoder_update, self.ht))
    self.h = self.decoder(decoder_update)

# Not "Defense of the Ancient"
class DotAttentionLayer(chainer.Chain):
  def __call__(self, S, h):
    return F.squeeze(F.softmax(F.batch_matmul(S, h)), axis=2)

class GeneralAttentionLayer(chainer.Chain):
  def __init__(self, hidden_size):
    super(GeneralAttentionLayer, self).__init__(
      inner_weight = chainer.links.Linear(hidden_size, hidden_size)
    )

  def __call__(self, S, h):
    batch_size, src_len, hidden_size = S.data.shape
    S = self.inner_weight(F.reshape(S, (batch_size * src_len, hidden_size)))
    S = F.reshape(S, (batch_size, src_len, hidden_size))
    a = F.softmax(F.squeeze(F.batch_matmul(S, h), axis = 2))
    return a

# MLP layer, as of Bahdanau+ 15
class MLPAttentionLayer(chainer.Chain):
  def __init__(self, hidden_size):
    super(MLPAttentionLayer, self).__init__(
      first_layer = chainer.links.Linear(2*hidden_size, hidden_size),
      second_layer = chainer.links.Linear(hidden_size, 1)
    )

  def __call__(self, S, h):
    batch_size, src_len, hidden_size = S.data.shape
    h = F.broadcast_to(F.expand_dims(h, axis=2), (batch_size, hidden_size, src_len))
    h = F.swapaxes(h, 1, 2)
    S = F.reshape(F.concat((S, h), axis=2), (batch_size * src_len, 2 * hidden_size))
    a = F.softmax(F.reshape(self.second_layer(F.tanh(self.first_layer(S))), (batch_size, src_len)))
    return a

# MISC class for holding the output
class Output(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

