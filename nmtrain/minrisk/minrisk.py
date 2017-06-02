import chainer
import numpy
import nmtrain
import math

from collections import defaultdict
from nmtrain.evals import bleu
from chainer.functions import copy, concat, expand_dims
from chainer.functions import select_item, get_item, swapaxes, forget
from chainer.functions import transpose, where, squeeze
from chainer.functions import log, softmax, exp

class MinimumRiskTraining(object):
  def __init__(self, minrisk_config, discriminator_loss=None):
    self.num_sample = minrisk_config.sampling_frequency
    self.generation_limit = minrisk_config.generation_limit
    self.sharpness = minrisk_config.sharpness
    self.eval_type = minrisk_config.eval_type
    self.is_discriminator = minrisk_config.eval_type == "discriminator"
    if minrisk_config.eval_type == "bleu":
      self.loss = lambda sample, y_t: \
                    -bleu.calculate_bleu_sentence_fast(sample, y_t,
                                                       ngram  = minrisk_config.bleu.ngram,
                                                       smooth = minrisk_config.bleu.smooth)
    elif minrisk_config.eval_type == "discriminator":
      self.loss = discriminator_loss
    else:
      raise ValueError("Unimplemented eval type for min-risk:", minrisk_config.eval_type)

  def __call__(self, model, src_batch, trg_batch, eos_id, outputer=None):
    h = model.encoder(src_batch)
    batch_size = trg_batch.shape[1]
    unique = defaultdict(set)

    sample_index  = numpy.zeros((self.num_sample, batch_size), dtype=bool)
    deltas, probs = [], []
    # Sampling
    for i in range(self.num_sample):
      delta, prob = self.sample(i, trg_batch, sample_index[i], model, h, unique, eos_id)
      deltas.append(delta)
      probs.append(prob * self.sharpness)

    # Calculate Risk + remove duplication
    risk = 0
    probs = concat(probs, axis=1)
    delta = numpy.concatenate(deltas, axis=1)
    sample_index = sample_index.transpose()
    for i in range(probs.shape[0]):
      prob = expand_dims(get_item(probs, i), axis=1)
      item = list(numpy.where(sample_index[i])[0])
      unique_prob = get_item(prob, [item])
      unique_prob = squeeze(softmax(transpose(unique_prob)), axis=0)
      valid_delta = chainer.Variable(model.xp.array(delta[i][item], dtype=numpy.float32))
      risk += chainer.functions.sum(unique_prob * valid_delta) / len(item)
    return risk / probs.shape[0]

  def sample(self, sample_num, trg_batch, sample_index, model, h, unique, eos_id):
    batch_size  = trg_batch.shape[1]
    sample_prob = 0
    if self.is_discriminator:
        # Ones is for the eos
        sample = numpy.ones((self.generation_limit, batch_size), dtype=numpy.int32)
    else:
        sample = numpy.zeros((self.generation_limit, batch_size), dtype=numpy.int32)
    end_flag    = None

    # Generate sample sentence
    model.decoder.init(h)
    for j in range(self.generation_limit):
      output = model.decode()
      y = softmax(output.y)
      if sample_num == 0:
        sample[j] = trg_batch[j]
      else:
        sample[j] = self.sample_one(y, sample[j-1] if j != 0 else numpy.zeros(len(sample[j])))
      next_word = chainer.Variable(model.xp.array(sample[j], dtype=numpy.int32))
      prob = select_item(log(y), next_word)
      sample_prob += prob

      # check_for_end:
      flag = sample[j] == eos_id
      if end_flag is None:
        end_flag = flag
      else:
        end_flag = numpy.logical_or(end_flag, flag)
      if numpy.all(end_flag):
        break
      else:
        model.update(next_word)
    delta = numpy.zeros(batch_size)
    if self.is_discriminator:
      delta = self.loss(sample.transpose())
      for i, sentence in enumerate(sample.transpose()):
        hash_value = tuple(sentence)
        if not hash_value in unique[i]:
          unique[i].add(hash_value)
          sample_index[i] = True
    else:
      # Calculate Risk
      for i, (sampled_sent, reference) in enumerate(zip(sample.transpose(), trg_batch.transpose())):
        # Cut the sent to first non zero
        idx  = numpy.where(sampled_sent == eos_id)[0]
        if len(idx) > 0:
          sent = tuple(sampled_sent[:idx[0] + 1])
        else:
          sent = tuple(sampled_sent)
        # Check for duplication
        hash_value = hash(sent)
        if not hash_value in unique[i]:
          unique[i].add(hash_value)
          sample_index[i] = True
          delta[i] = self.loss(sent, tuple(reference))
        else:
          delta[i] = float("inf")

    return numpy.expand_dims(delta, axis=1),\
           expand_dims(sample_prob, axis=1)

  def sample_one(self, probs_var, last_sample):
    probs = copy(probs_var, -1).data
    if self.is_discriminator:
      samples = numpy.ones(len(probs))
    else:
      samples = numpy.zeros(len(probs))
    for i, prob in enumerate(probs):
      if last_sample[i] == 1:
        samples[i] == 1
      else:
        samples[i] = numpy.random.choice(len(prob), p=prob)
    return samples

