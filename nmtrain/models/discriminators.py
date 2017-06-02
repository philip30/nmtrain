import chainer
import numpy
import nmtrain
import math

from chainer.functions import sigmoid
from chainer.functions import max_pooling_2d

class Conv2DTargetDiscriminator(chainer.Chain):
  def __init__(self, hidden_units, dropout, embedding_size, generation_limit):
    super(Conv2DTargetDiscriminator, self).__init__()
    # Init all the links
        # Calculate perceptron layer size manually
    h0 = 1 + (embedding_size - hidden_units.filter_size.x)
    w0 = 1 + (generation_limit - hidden_units.filter_size.y)
    h0 = math.ceil(1 + (h0 - hidden_units.max_pool_size.x) / hidden_units.max_pool_size.x)
    w0 = math.ceil(1 + (w0 - hidden_units.max_pool_size.y) / hidden_units.max_pool_size.y)
    mid_size = int(h0 * w0 * hidden_units.output_channel)
    self.max_pool_size = (hidden_units.max_pool_size.x, hidden_units.max_pool_size.y)
    self.dropout = dropout
    with self.init_scope:
      self.conv1 = chainer.links.Convolution2D(hidden_units.input_channel,
                                               hidden_units.output_channel,
                                               (hidden_units.filter_size.x, hidden_units.filter_size.y))
      self.percept = chainer.links.Linear(mid_size, 2)

  def __call__(self, embeddings):
    # Generator
    h1 = chainer.functions.tanh(self.conv1(embeddings))
    h1 = max_pooling_2d(h1, self.max_pool_size)
    h1 = chainer.functions.dropout(h1, ratio=self.dropout.percept_layer, train=is_train)
    return self.percept(h1)

