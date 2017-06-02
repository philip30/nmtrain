import chainer
import numpy
import nmtrain

from chainer.links import EmbedID
from chainer.links import Linear
from chainer.functions import dropout
from chainer.functions import batch_matmul
from chainer.functions import concat
from chainer.functions import expand_dims
from chainer.functions import squeeze
from chainer.functions import swapaxes
from chainer.functions import concat
from chainer.functions import tanh
from nmtrain.components import StackLSTM

class BidirectionalAttentionalEncoder(chainer.Chain):
  def __init__(self, in_size, hidden_units, dropouts, lstm_depth, input_feeding=True, lexicon=None):
    super(BidirectionalAttentionalEncoder, self).__init__()
    E = hidden_units.embed
    H = hidden_units.stack_lstm
    D = lstm_depth
    self.dropouts = dropouts
    self.lexicon  = lexicon
    # Links
    with self.init_scope():
      self.embed = EmbedID(in_size, E)
      self.encode_forward = StackLSTM(E, H, lstm_depth, dropouts.stack_lstm)
      self.encode_backward = StackLSTM(E, H, lstm_depth, dropouts.stack_lstm)
      self.encode_project = Linear(2 * H, H)

  def __call__(self, src_data):
    embed_dropout = lambda link: dropout(link, ratio=self.dropouts.encode_embed)
    encode_dropout = lambda link: dropout(link, ratio=self.dropouts.encode)
    # Reset both encoders
    self.encode_forward.reset_state(None)
    self.encode_backward.reset_state(None)

    # Perform encoding
    fe, be = [], []
    src_input = self.xp.array(src_data, dtype=numpy.int32)
    for j in range(len(src_input)):
      fe.append(self.encode_forward(embed_dropout(self.embed(chainer.Variable(src_input[j])))))
      be.append(self.encode_backward(embed_dropout(self.embed(chainer.Variable(src_input[-j-1])))))

    # Joining encoding together
    S = []
    for j in range(len(fe)):
      h = encode_dropout(self.encode_project(concat((fe[j], be[-1-j]), axis=1)))
      S.append(expand_dims(h, axis=2))
    S = swapaxes(concat(S, axis=2), 1, 2)

    # If lexicon is provided
    if self.lexicon is not None:
      lex_matrix = chainer.Variable(self.xp.array(self.lexicon.init(src_data), dtype=numpy.float32))
    else:
      lex_matrix = None

    return h, S, lex_matrix

