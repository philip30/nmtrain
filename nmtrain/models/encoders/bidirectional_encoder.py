import chainer
import numpy
import nmtrain

from chainer.links import EmbedID
from chainer.links import Linear
from chainer.functions import dropout
from chainer.functions import concat
from nmtrain.components import StackLSTM

class BidirectionalEncoder(chainer.Chain):
  def __init__(self, in_size, hidden_units, dropouts, lstm_depth):
    super(BidirectionalEncoder, self).__init__()
    E = hidden_units.embed
    H = hidden_units.stack_lstm
    D = lstm_depth
    self.dropouts = dropouts

    with self.init_scope:
      self.embed = EmbedID(in_size, E)
      self.encode_forward = StackLSTM(E, H, D, dropouts.stack_lstm)
      self.encode_backward = StackLSTM(E, H, D, dropouts.stack_lstm)
      self.encode_project = Linear(2 * H, H)


  def __call__(self, src_data):
    # The dropout function
    embed_dropout = lambda link: dropout(link, ratio=self.dropouts.encode_embed)
    # Reset both encoders
    self.encode_forward.reset_state(None)
    self.encode_backward.reset_state(None)

    # Perform encoding
    src_sent = self.xp.array(src_data, dtype=numpy.int32)
    for j in range(len(src_sent)):
      fe = self.encode_forward(embed_dropout(self.embed(chainer.Variable(src_sent[j]))))
      be = self.encode_backward(embed_dropout(self.embed(chainer.Variable(src_sent[-j-1]))))
    encoded = concat((fe,be), axis=1)
    return dropout(self.encode_project(encoded), ratio=self.dropouts.encode)

