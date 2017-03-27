import chainer
import chainer.functions as F
import numpy

import nmtrain

class RNNTargetDiscriminator(chainer.Chain):
  def __init__(self, embed_size, hidden_size, dropout_ratio):
    super(RNNTargetDiscriminator, self).__init__()
    self.add_link("hidden", nmtrain.chner.StackLSTM(hidden_size, hidden_size, 1, dropout_ratio))
    self.add_link("assess", chainer.links.Linear(hidden_size, 2))

  def __call__(self, src_batch, trg_batch, target_embedding, train_discriminator = True):
    # Generator
    embed_volatility = "on" if train_discriminator else "off"

    self.hidden.reset_state()
    embedded = []
    for data in trg_batch:
      embed_data = target_embedding(chainer.Variable(self.xp.array(data, dtype=numpy.int32), volatile=embed_volatility))
      if train_discriminator:
        # Freeze generator
        embed_data.volatile = "off"
      else:
        # Freeze discriminator
        embed_data.volatile = "on"
      state = self.hidden(embed_data)
    return self.assess(F.tanh(state))

