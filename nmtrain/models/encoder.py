import chainer
import chainer.functions as F
import numpy

import nmtrain
import nmtrain.chner

class BidirectionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(BidirectionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(2*hidden_size, hidden_size)
    )
    self.dropout_ratio = dropout_ratio

  def __call__(self, src_data):
    # The dropout function
    dropout = lambda link: F.dropout(link, ratio=self.dropout_ratio, train=nmtrain.environment.is_train())
    mem_optimize = nmtrain.optimization.chainer_mem_optimize
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    for j in range(len(src_data)):
      forward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_data[j]), level=1))
      backward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_data[-j-1]), level=1))
      fe = self.encode_forward(forward_embed)
      be = self.encode_backward(backward_embed)

    return dropout(mem_optimize(self.encode_project, F.concat((fe, be), axis=1), level=2))

class BidirectionalAttentionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth, input_feeding=True):
    super(BidirectionalAttentionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(2*hidden_size, embed_size)
    )
    self.input_feeding = input_feeding
    self.dropout_ratio = dropout_ratio

  def __call__(self, src_data):
    # The dropout function
    dropout = lambda link: F.dropout(link, ratio=self.dropout_ratio, train=nmtrain.environment.is_train())
    mem_optimize = nmtrain.optimization.chainer_mem_optimize
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    fe, be = [], []
    for j in range(len(src_data)):
      forward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_data[j]), level=1))
      backward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_data[-j-1]), level=1))
      fe.append(self.encode_forward(forward_embed))
      be.append(self.encode_backward(backward_embed))

    # Joining encoding together
    S = []
    for j in range(len(fe)):
      h = mem_optimize(self.encode_project, F.concat((fe[j], be[-1-j]), axis=1), level=2)
      S.append(F.expand_dims(h, axis=2))
    S = F.swapaxes(F.concat(S, axis=2), 1, 2)

    # Append 0 to the end of h
    if self.input_feeding:
      xp = nmtrain.environment.array_module()
      shape = fe[0].data.shape # (batch_size, hidden_size)
      zero = nmtrain.environment.Variable(xp.zeros(shape, dtype=numpy.float32))
      h = F.hstack((h, zero))

    return h, S

