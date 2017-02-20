import chainer
import numpy

import nmtrain

class BidirectionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(BidirectionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(hidden_size, hidden_size)
    )
    self.dropout_ratio = dropout_ratio

  def __call__(self, src_data):
    # The dropout function
    dropout = lambda link: chainer.functions.dropout(link, ratio=self.dropout_ratio, train=nmtrain.environment.is_train())
    mem_optimize = nmtrain.optimization.chainer_mem_optimize
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    src_sent = self.xp.array(src_data, dtype=numpy.int32)
    for j in range(len(src_sent)):
      forward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_data[j]), level=1))
      backward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_data[-j-1]), level=1))
      fe = self.encode_forward(forward_embed)
      be = self.encode_backward(backward_embed)

    return dropout(self.encode_project(fe) + be)

