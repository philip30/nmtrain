import chainer
import numpy

import nmtrain

class BidirectionalAttentionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth,
               input_feeding=True, lexicon=None):
    super(BidirectionalAttentionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(2 * hidden_size, embed_size),
    )
    self.dropout_ratio = dropout_ratio
    self.lexicon       = lexicon

  def __call__(self, src_data):
    # Some function naming
    F = chainer.functions
    dropout = lambda link: F.dropout(link, ratio=self.dropout_ratio, train=nmtrain.environment.is_train())
    mem_optimize = nmtrain.optimization.chainer_mem_optimize
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    fe, be = [], []
    src_input = self.xp.array(src_data, dtype=numpy.int32)
    for j in range(len(src_input)):
      forward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_input[j]), level=1))
      backward_embed = dropout(mem_optimize(self.embed, nmtrain.environment.Variable(src_input[-j-1]), level=1))
      fe.append(self.encode_forward(forward_embed))
      be.append(self.encode_backward(backward_embed))

    # Joining encoding together
    S = []
    for j in range(len(fe)):
      h = self.encode_project(F.concat((fe[j], be[-1-j]), axis=1))
      S.append(F.expand_dims(h, axis=2))
    S = F.swapaxes(F.concat(S, axis=2), 1, 2)

    # If lexicon is provided
    if self.lexicon is not None:
      lex_matrix = nmtrain.environment.Variable(self.lexicon.init(src_data, self.xp))
    else:
      lex_matrix = None

    return h, S, lex_matrix

