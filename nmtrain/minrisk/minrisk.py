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
    batch_size = src_batch.shape[1]
    unique = defaultdict(set)

    sample_index = numpy.zeros((self.num_sample, batch_size), dtype=bool)
    probs = []
    deltas = numpy.zeros((self.num_sample, batch_size), dtype=numpy.float32)
    # Note that ones represent discriminator
    samples = []
    # Sampling
    for i in range(self.num_sample):
      prob, sample = self.sample(i, trg_batch, sample_index[i], model, h, unique, eos_id, batch_size, deltas[i])
      probs.append(prob * self.sharpness)
      samples.append(sample)
    
    # Calculate Risk + remove duplication
    risk = 0
    probs = concat(probs, axis=1)
    sample_index = sample_index.transpose()
    deltas = deltas.transpose()
    if outputer:
      outputer.begin_collection(src=src_batch, ref=trg_batch)

    for i in range(probs.shape[0]):
      prob = expand_dims(get_item(probs, i), axis=1)
      item = list(numpy.where(sample_index[i])[0])
      unique_prob = get_item(prob, [item])
      unique_prob = squeeze(softmax(transpose(unique_prob)), axis=0)
      valid_delta = model.xp.asarray(deltas[i][item], dtype=numpy.float32)
      if outputer:
        outputer(nmtrain.data.Data(minrisk_prob=unique_prob, minrisk_delta=valid_delta, minrisk_item=item))

      risk += chainer.functions.sum(unique_prob * valid_delta) / unique_prob.shape[0]

    if outputer:
      outputer(nmtrain.data.Data(minrisk_sample=samples))
      outputer.end_collection()

    return risk / probs.shape[0]

  def sample(self, sample_num, trg_batch, sample_index, model, h, unique, eos_id, batch_size, delta):
    sample_prob = 0
    end_flag    = None
    flag = None

    sample = []
    # Generate sample sentence
    model.decoder.init(h)
    for j in range(self.generation_limit):
      output = model.decode()
      y = softmax(output.y)
      if sample_num == 0 and trg_batch is not None:
        sample.append(trg_batch[j])
      else:
        if len(sample) > 1:
          last_sample = sample[-1]
        else:
          last_sample = None
        sample.append(self.sample_one(y, flag, eos_id))
      next_word = model.xp.array(sample[j], dtype=numpy.int32)
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
    sample = numpy.array(sample, dtype=int).transpose()
    if self.is_discriminator:
      delta[:] = self.loss(sample)
      for i, sentence in enumerate(sample):
        hash_value = tuple(sentence)
        if not hash_value in unique[i]:
          unique[i].add(hash_value)
          sample_index[i] = True
    else:
      # Calculate Risk
      for i, (sampled_sent, reference) in enumerate(zip(sample, trg_batch.transpose())):
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

    return expand_dims(sample_prob, axis=1), sample

  def sample_one(self, probs_var, flag, eos_id):
    probs = copy(probs_var, -1).data
    samples = numpy.zeros(len(probs), dtype=int)
    eos_index = list(numpy.where(flag))
    for i, prob in enumerate(probs):
      samples[i] = numpy.random.choice(len(prob), p=prob)

      if len(eos_index) != 0:
        samples[eos_index] = eos_id
    return samples

