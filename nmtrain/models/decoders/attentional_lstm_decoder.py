import chainer
import nmtrain

from chainer.links import Linear
from chainer.links import EmbedID
from chainer.functions import squeeze
from chainer.functions import batch_matmul
from chainer.functions import tanh
from chainer.functions import concat, forget
from nmtrain.components import StackLSTM
from nmtrain.models import attentions
from nmtrain.models import lexicons

# Implementation of Luong et al.
class LSTMAttentionalDecoder(chainer.Chain):
  def __init__(self, out_size, hidden_units,
               dropouts, lstm_depth, input_feeding,
               attention_type, lexicon):
    super(LSTMAttentionalDecoder, self).__init__()
    # Construct Appropriate Attention Chain
    if attention_type == "dot":
      attention = attentions.DotAttentionLayer()
    elif attention_type == "general":
      attention = attentions.GeneralAttentionLayer(hidden_units.stack_lstm)
    elif attention_type == "mlp":
      attention = attentions.MLPAttentionLayer(hidden_units.stack_lstm, hidden_units.attention)
    else:
      raise ValueError("Unknown Attention Type:", attention_type)

    # Construct Appropriate Lexicon Chain
    if lexicon is not None:
      if lexicon.type == "bias":
        lexicon_model = lexicons.BiasedLexicon(lexicon.alpha)
      else:
        raise ValueError("Unknown Lexicon Type:", lexicon.type)

    # Register all 
    E = hidden_units.embed
    H = hidden_units.stack_lstm
    D = lstm_depth
    self.input_feeding = input_feeding
    self.dropouts      = dropouts

    ### Chainer Registration
    with self.init_scope():
      self.decoder = StackLSTM(E, H, D, dropouts.stack_lstm)
      self.context_project = Linear(2 * H, H)
      self.affine_vocab = Linear(H, out_size)
      self.output_embed = EmbedID(out_size, E)
      self.attention = attention

      if lexicon is not None:
        self.lexicon_model = lexicon_model

      if input_feeding:
        self.feeding_transform = chainer.links.Linear(H+E, E)
    ### End Chainer Registration

  def init(self, h):
    h, S, lexicon_matrix = h
    self.decoder.reset_state(h)
    self.h = h
    self.S = S
    self.lexicon_matrix = lexicon_matrix

  def __call__(self):
    # Calculate Attention vector
    a = self.attention(self.S, self.h)
    # Calculate context vector
    c = squeeze(batch_matmul(self.S, a, transa=True), axis=2)
    # Calculate hidden vector + context
    self.ht = self.context_project(concat((self.h, c), axis=1))
    # Calculate Word probability distribution
    y = self.affine_vocab(tanh(self.ht))
    if self.lexicon_matrix is not None:
      y = self.lexicon_model(y, a, self.ht, self.lexicon_matrix)

    return nmtrain.data.Data(y=y, a=a)

  def update(self, next_word):
    # embed_size + hidden size -> input feeding approach
    decoder_update = self.output_embed(next_word)
    if self.input_feeding:
      decoder_update = self.feeding_transform(concat((self.ht, decoder_update), axis=1))
    self.h = self.decoder(decoder_update)
    return decoder_update, self.h

  def set_state(self, state):
    self.h, self.ht, state = state
    self.decoder.set_state(state)

  def state(self):
    return self.h, self.ht, self.decoder.state()

