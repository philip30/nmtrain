import chainer
import chainer.functions as F

import nmtrain

class BiasedLexicon(chainer.Chain):
  def __init__(self, alpha):
    super(BiasedLexicon, self).__init__()
    self.alpha = alpha

  def __call__(self, y, a, ht, y_lex):
    y_dict = F.squeeze(F.batch_matmul(y_lex, a, transa=True), axis=2)
    return (y + F.log(y_dict + self.alpha))

#class LinearInterpolationLexicon(chainer.Chain):
#  def __init__(self, hidden_size):
#    super(LinearInterpolationLexicon, self).__init__(
#      perceptron = chainer.links.Linear(hidden_size, 1)
#    )
#
#  def __call__(self, y, a, ht, y_lex):
#    y      = F.softmax(y)
#    y_dict = F.squeeze(F.batch_matmul(y_lex, a, transa=True), axis=2)
#    gamma  = F.broadcast_to(F.sigmoid(self.perceptron(ht)), y_dict.data.shape)
#    return (gamma * y_dict + (1-gamma) * y)
#
