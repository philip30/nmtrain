import chainer
import numpy
import nmtrain
import math

from chainer.functions import sigmoid
from chainer.functions import squeeze
from chainer.functions import max_pooling_2d

class Conv2DTargetDiscriminator(chainer.Chain):
  def __init__(self, hidden_units, dropout, embedding_size):
    super(Conv2DTargetDiscriminator, self).__init__()
    # Init all the links
    # Calculate perceptron layer size manually
    self.dropout = dropout
    with self.init_scope():
      self.conv1 = chainer.links.Convolution2D(1, hidden_units.feature_size, (embedding_size, hidden_units.ngram))
      self.percept = chainer.links.Linear(hidden_units.feature_size, 2)

  def __call__(self, embeddings):
    # Generator
    h1 = chainer.functions.tanh(self.conv1(embeddings))
    # Max pooling over time from all the filter layer
    h1 = max_pooling_2d(h1, embeddings.shape[3]-2)
    h1 = chainer.functions.dropout(h1, ratio=self.dropout.percept_layer)
    return self.percept(h1)

