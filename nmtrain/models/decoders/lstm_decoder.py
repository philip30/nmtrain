import chainer
import nmtrain

from chainer.links import EmbedID
from chainer.links import Linear
from chainer.functions import dropout
from nmtrain.components import StackLSTM

class LSTMDecoder(chainer.Chain):
  def __init__(self, out_size, hidden_units, dropouts, lstm_depth):
    super(LSTMDecoder, self).__init__()
    E = hidden_units.embed
    H = hidden_units.stack_lstm
    D = lstm_depth
    self.dropouts = dropouts
    ### Chainer Registration
    with self.init_scope:
      self.decoder = StackLSTM(E, H, D, dropouts.stack_lstm)
      self.affine_vocab = Linear(H, out_size)
      self.output_embed = EmbedID(out_size, E)
    ### End Chainer registration

  def __call__(self):
    y =  self.affine_vocab(chainer.functions.tanh(self.h))
    return nmtrain.models.decoders.Output(y=y)

  def init(self, h):
    self.h = h
    self.decoder.reset_state(h)

  def update(self, next_word):
    self.h = self.decoder(self.output_embed(next_word))
    return self.h

  def set_state(self, state):
    self.h, state = state
    self.decoder.set_state(state)

  def state(self):
    return self.h, self.decoder.state()

