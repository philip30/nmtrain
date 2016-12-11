import chainer
import chainer.functions as F

import nmtrain

class BiasedLexicon(chainer.Chain):
  def __init__(self, lexicon):
    super(BiasedLexicon, self).__init__()
    self.lexicon = lexicon

  def __call__(self, y, a, ht):
    y_lex = self.lexicon.p_lex()
    y_dict = F.squeeze(F.batch_matmul(y_lex, a, transa=True), axis=2)
    return (y + F.log(y_dict + self.lexicon.alpha)), False

class LinearInterpolationLexicon(chainer.Chain):
  def __init__(self, lexicon, hidden_size):
    super(LinearInterpolationLexicon, self).__init__(
      perceptron = chainer.links.Linear(hidden_size, 1)
    )
    self.lexicon = lexicon

  def __call__(self, y, a, ht):
    y      = F.softmax(y)
    y_lex  = self.lexicon.p_lex()
    y_dict = F.squeeze(F.batch_matmul(y_lex, a, transa=True), axis=2)
    gamma  = F.broadcast_to(F.sigmoid(self.perceptron(ht)), y_dict.data.shape)
    return (gamma * y_dict + (1-gamma) * y), True

class IdentityLexicon(chainer.Chain):
  def __call__(self, y, a, ht):
    return y, False

