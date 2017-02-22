import chainer
import chainer.functions as F
import numpy

import nmtrain

class RNNTargetDiscriminator(chainer.Chain):
  def __init__(self, embed_size, hidden_size, dropout_ratio):
    super(RNNTargetDiscriminator, self).__init__()
    self.add_link("hidden", nmtrain.chner.StackLSTM(embed_size, hidden_size, 1, dropout_ratio))
    self.add_link("assess", chainer.links.Linear(hidden_size, 1))

  def discriminate_target(self, trg_data, target_embedding):
    self.hidden.reset_state()
    embedded = []
    for data in trg_data:
      embed_data = target_embedding(chainer.Variable(self.xp.array(data, dtype=numpy.int32), volatile='on'))
      if nmtrain.environment.is_train():
        embed_data.volatile = "off"
      state = self.hidden(embed_data)
    discriminator_value = self.assess(F.tanh(F.dropout(state)))
    return F.sigmoid(discriminator_value)

