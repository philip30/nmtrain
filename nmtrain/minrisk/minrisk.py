import chainer
import numpy
import math

from collections import defaultdict
from nmtrain.evals import bleu
from chainer.functions import copy
from chainer.functions import softmax
from chainer.functions import select_item
from chainer.functions import expand_dims
from chainer.functions import hstack
from chainer.functions import get_item
from chainer.functions import transpose
from chainer.functions import log, softmax, exp

class MinimumRiskTraining(object):
  def __init__(self, minrisk_config):
    self.num_sample = minrisk_config.sampling_frequency
    self.generation_limit = minrisk_config.generation_limit
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

  def __call__(self, model, src_batch, trg_batch, eos_id, outputer=None, is_train=True):
    volatile = chainer.OFF if is_train else chainer.ON
    model.encode(src_batch)

    if outputer: outputer.begin_collection(src=src_batch, ref=trg_batch)
    samples = numpy.zeros((self.num_sample, self.generation_limit, src_batch.shape[1]), dtype=numpy.int32)
    sample_probs = None
    # Sampling
    for i in range(self.num_sample):
      sample_prob = 0
      end_flag    = None
      for j in range(self.generation_limit):
        output = model.decode()
        y = softmax(output.y)
        if i == 0:
          samples[i][j] = trg_batch[j]
        else:
          samples[i][j] = self.sample(y)
        next_word = chainer.Variable(model.xp.array(samples[i][j], dtype=numpy.int32), volatile=volatile)
        sample_prob += select_item(chainer.functions.log(y), next_word) * self.sharpness

        # check_for_end:
        flag = samples[i][j] == eos_id
        if end_flag is None:
          end_flag = flag
        else:
          end_flag = numpy.logical_or(end_flag, flag)
        # Note: EOF has 0 id.
        if numpy.all(end_flag):
          break
        else:
          model.update(next_word)
        if i == 0 and outputer: outputer(output)
      sample_prob = expand_dims(sample_prob, axis=1)
      if sample_probs is None:
        sample_probs = sample_prob
      else:
        sample_probs = hstack((sample_probs, sample_prob))
    if outputer: outputer.end_collection()
    sample_probs = transpose(softmax(sample_probs))

    # Calculate Risk
    unique = defaultdict(set)
    risk   = 0
    for i, sample in enumerate(samples):
      sample_prob = get_item(sample_probs, i)
      delta = numpy.zeros(src_batch.shape[1])
      sample_index = []
      for j, (sampled_sent, reference) in enumerate(zip(sample.transpose(), trg_batch.transpose())):
        # Cut the sent to first non zero
        idx  = numpy.where(sampled_sent == 0)[0]
        if len(idx) > 0:
          sent = tuple(sampled_sent[:idx[0] + 1])
        else:
          sent = tuple(sampled_sent)
        # Check for duplication
        hash_value = hash(sent)
        if not hash_value in unique[i]:
          unique[i].add(hash_value)
          sample_index.append(j)
          delta[j] = self.loss(sent, tuple(reference))
      delta = numpy.broadcast_to(delta, (len(sample_index), delta.shape[0]))
      sample_prob = chainer.functions.broadcast_to(sample_prob, (len(sample_index), sample_prob.shape[0]))
      items = chainer.Variable(model.xp.array(sample_index, dtype=numpy.int32), volatile=volatile)
      delta = select_item(chainer.Variable(model.xp.array(delta, dtype=numpy.float32), volatile=volatile), items)
      prob  = select_item(sample_prob, items)
      risk += chainer.functions.sum(delta * prob) / len(items.data)

    return risk / len(samples)

  def sample(self, probs_var):
    probs = copy(probs_var, -1).data
    samples = numpy.zeros(len(probs))
    for i, prob in enumerate(probs):
      samples[i] = numpy.random.choice(len(prob), p=prob)
    return samples

