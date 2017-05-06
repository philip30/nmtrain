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
from nmtrain.chner import StackLSTM

class BidirectionalAttentionalEncoder(chainer.Chain):
  def __init__(self, in_size, hidden_units, dropouts, lstm_depth, input_feeding=True, lexicon=None):
    super(BidirectionalAttentionalEncoder, self).__init__()
    E = hidden_units.embed
    H = hidden_units.stack_lstm
    D = lstm_depth
    # Links
    self.add_link("embed", EmbedID(in_size, E))
    self.add_link("encode_forward", StackLSTM(E, H, lstm_depth, dropouts.stack_lstm))
    self.add_link("encode_backward", StackLSTM(E, H, lstm_depth, dropouts.stack_lstm))
    self.add_link("encode_project", Linear(2 * H, E))
    self.add_link("encode_init", Linear(H, E))
    # Attributes
    self.dropouts = dropouts
    self.lexicon  = lexicon

  def __call__(self, src_data, is_train):
    embed_dropout = lambda link: dropout(link, ratio=self.dropouts.encode_embed, train=is_train)
    encode_dropout = lambda link: dropout(link, ratio=self.dropouts.encode, train=is_train)
    volatile = chainer.OFF if is_train else chainer.ON
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    fe, be = [], []
    src_input = self.xp.array(src_data, dtype=numpy.int32)
    for j in range(len(src_input)):
      fe.append(self.encode_forward(embed_dropout(self.embed(chainer.Variable(src_input[j], volatile=volatile))), is_train))
      be.append(self.encode_backward(embed_dropout(self.embed(chainer.Variable(src_input[-j-1], volatile=volatile))), is_train))

    # Joining encoding together
    S = []
    for j in range(len(fe)):
      h = encode_dropout(self.encode_project(concat((fe[j], be[-1-j]), axis=1)))
      S.append(expand_dims(h, axis=2))
    S = swapaxes(concat(S, axis=2), 1, 2)

    h = encode_dropout(self.encode_init(h))

    # If lexicon is provided
    if self.lexicon is not None:
      lex_matrix = chainer.Variable(self.xp.array(self.lexicon.init(src_data), dtype=numpy.float32), volatile=volatile)
    else:
      lex_matrix = None

    return h, S, lex_matrix

