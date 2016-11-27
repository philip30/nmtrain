import chainer
import chainer.functions as F
import numpy

import nmtrain
import nmtrain.chner
import nmtrain.environment

class BidirectionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth, attention=False):
    super(BidirectionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(2*hidden_size, embed_size)
    )
    self.use_attention = attention

  def __call__(self, src_data):
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    for j in range(len(src_data)):
      fe = self.encode_forward(self.embed(nmtrain.environment.Variable(src_data[j])))
      be = self.encode_backward(self.embed(nmtrain.environment.Variable(src_data[-j-1])))

    # TODO(philip30): Implement attention
    return self.encode_project(F.concat((fe, be), axis=1))

