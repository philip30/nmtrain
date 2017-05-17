import chainer
import nmtrain

from chainer.links import EmbedID
from chainer.links import Linear
from chainer.functions import dropout
from nmtrain.chner import StackLSTM

class LSTMDecoder(chainer.Chain):
  def __init__(self, out_size, hidden_units, dropouts, lstm_depth):
    super(LSTMDecoder, self).__init__()
    E = hidden_units.embed
    H = hidden_units.stack_lstm
    D = lstm_depth
    # Links
    self.add_link("decoder", StackLSTM(E, H, D, dropouts.stack_lstm))
    self.add_link("affine_vocab", Linear(H, out_size))
    self.add_link("output_embed", EmbedID(out_size, E))
    # Attributes
    self.dropouts = dropouts

  def __call__(self, is_train):
    y =  self.affine_vocab(chainer.functions.tanh(self.h))
    return nmtrain.models.decoders.Output(y=y)

  def init(self, h, is_train):
    self.h = h
    self.decoder.reset_state(h)

  def update(self, next_word, is_train):
    self.h = self.decoder(self.output_embed(next_word), is_train)
    return self.h

  def set_state(self, state):
    self.h, state = state
    self.decoder.set_state(state)

  def state(self):
    return self.h, self.decoder.state()

