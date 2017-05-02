import chainer
import numpy
import math

from nmtrain.evals import bleu
from chainer.functions import copy
from chainer.functions import softmax
from chainer.functions import select_item
from chainer.functions import expand_dims
from chainer.functions import hstack
from chainer.functions import log, softmax, exp

class MinimumRiskTraining(object):
  def __init__(self, minrisk_config):
    self.num_sample = minrisk_config.sampling_frequency
    self.sharpness  = minrisk_config.sharpness
    if minrisk_config.eval_type == "bleu":
      self.loss = lambda sample, y_t: \
                    -bleu.calculate_bleu_sentence_fast(sample, y_t,
                                                       ngram  = minrisk_config.bleu.ngram,
                                                       smooth = minrisk_config.bleu.smooth)
    else:
      raise ValueError("Unimplemented eval type for min-risk:", minrisk_config.eval_type)

  def set_train(self, is_train):
    self.is_train = is_train

  def __call__(self, y_var, y_t):
    xp = chainer.cuda.get_array_module(y_var, y_t)
    volatile = chainer.OFF if self.is_train else chainer.ON
    y_var = softmax(y_var)
    y = copy(y_var, -1).data
    sample = numpy.zeros((y.shape[0], self.num_sample), dtype=numpy.int32)
    for i in range(len(sample)):
      sample[i][:] = numpy.random.choice(len(y[i]), size=self.num_sample, p=y[i])
      sample[i][0] = y_t[i]

    log_prob = None
    for col_sample in sample.transpose():
      prob = expand_dims(select_item(y_var,
                                     chainer.Variable(xp.array(col_sample, dtype=numpy.int32),
                                                      volatile=volatile)), axis=1)
      prob = log(prob)
      if log_prob is None:
        log_prob = prob
      else:
        log_prob = chainer.functions.hstack((log_prob, prob))
    return sample, log_prob

  def calculate_risk(self, batch_sample, batch_reference, log_probs):
    xp = chainer.cuda.get_array_module(log_probs)
    risk = numpy.zeros((batch_sample.shape[0], self.num_sample), dtype=numpy.float32)
    volatile = chainer.OFF if self.is_train else chainer.ON
    for risk_i, samples, reference in zip(risk, batch_sample, batch_reference.transpose()):
      reference = tuple(reference)
      for j in range(self.num_sample):
        risk_i[j] = self.loss(tuple(samples[j]), reference)

    prob = softmax(log_probs * self.sharpness)
    risk = chainer.Variable(xp.array(risk, dtype=numpy.float32), volatile=volatile)
    return chainer.functions.sum(prob * risk) / batch_sample.shape[0]

