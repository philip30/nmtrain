import chainer
import chainer.functions as F
import numpy

import nmtrain

class RNNTargetDiscriminator(chainer.Chain):
  def __init__(self, embed_size, hidden_size, dropout_ratio):
    super(RNNTargetDiscriminator, self).__init__()
    self.add_link("hidden", nmtrain.chner.StackLSTM(embed_size, hidden_size, 1, dropout_ratio))
    self.add_link("assess", chainer.links.Linear(hidden_size, 2))

  def __call__(self, embeddings, train=True):
    # Generator
    self.hidden.reset_state()
    for embed in embeddings:
      if not train:
        embed.volatile = "on"
      else:
        embed.volatile = "off"
      state = self.hidden(embed)
    return self.assess(F.tanh(state))

