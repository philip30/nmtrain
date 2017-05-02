import numpy
import math
import functools

import nmtrain.util as util

class UnknownTrainer(object):
  def __init__(self, include_rare=True):
    self.include_rare = include_rare

  def include_rare(self):
    return self.include_rare

class UnknownNormalTrainer(UnknownTrainer):
  def __init__(self):
    super(UnknownNormalTrainer, self).__init__(False)

  def __iter__(self):
    yield lambda batch: batch.normal_data

class UnknownWordDropoutTrainer(UnknownTrainer):
  def __init__(self, corpus_divider=1):
    assert(corpus_divider > 0), "Invalid Gamma value."
    self.src_freq_map = None
    self.trg_freq_map = None
    self.corpus_divider = corpus_divider

  def dropout_word(self, batch, freq_map, total_count):
    flag = numpy.zeros_like(batch)
    for row in range(len(batch)):
      for col in range(len(batch[row])):
        word = batch[row][col]
        if word in freq_map:
          do_prob = self.dropout_prob(freq_map[word], total_count)
          if numpy.random.random() >= do_prob:
            flag[row][col] = 1
          else:
            flag[row][col] = 0
        else:
          flag[row][col] = 0
    return batch * flag
  
  @functools.lru_cache(maxsize=1)
  def src_sum(self):
    return sum(self.src_freq_map.values())
  
  @functools.lru_cache(maxsize=1)
  def trg_sum(self):
    return sum(self.trg_freq_map.values())
  
  # 2 MB cache
  @functools.lru_cache(maxsize=2 ** 20)
  def dropout_prob(self, word_freq, total_count):
    prob_not_seen = 1 - (word_freq / total_count)
    return math.exp(math.log(prob_not_seen) * total_count / self.corpus_divider)

  def __iter__(self):
    yield lambda batch: \
            (self.dropout_word(batch.normal_data[0], self.src_freq_map, self.src_sum()), \
             self.dropout_word(batch.normal_data[1], self.trg_freq_map, self.trg_sum()))

class UnknownSentenceDropoutTrainer(UnknownTrainer):
  def __init__(self, ratio=0.2):
    self.ratio = ratio

  def dropout_sentence(self, batch):
    odds = numpy.random.uniform(low = 0.0, high = 1.0)
    if odds > self.ratio:
      return batch.normal_data
    else:
      return batch.unk_data

  def __iter__(self):
    yield lambda batch: self.dropout_sentence(batch)

def from_config(config):
  method = config.method
  if method == "word_dropout":
    return UnknownWordDropoutTrainer(corpus_divider = config.corpus_divider)
  elif method == "sentence_dropout":
    return UnknownSentenceDropoutTrainer(ratio = config.dropout_ratio)
  elif method == "normal":
    return UnknownNormalTrainer()
  else:
    raise ValueError("Unknown unknown_training method:", method)

