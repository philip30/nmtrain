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

class UnknownRedundancyTrainer(UnknownTrainer):
  def __iter__(self):
    yield lambda batch: batch.normal_data
    yield lambda batch: batch.unk_data

class UnknownWordDropoutTrainer(UnknownTrainer):
  def __init__(self, gamma=5):
    assert(gamma > 0 and type(gamma) == int), "Invalid Gamma value."
    self.src_freq_map = None
    self.trg_freq_map = None
    self.gamma = gamma

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
    return math.exp(math.log(prob_not_seen) * total_count / self.gamma)

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

def from_string(string):
  col = string.split(":")
  method = col[0]
  if len(col) == 1:
    param_str = ""
  else:
    param_str = col[1]
  if method == "redundancy":
    return UnknownRedundancyTrainer()
  elif method == "word_dropout":
    param = util.parse_parameter(param_str, {"gamma": int})
    return UnknownWordDropoutTrainer(**param)
  elif method == "sentence_dropout":
    param = util.parse_parameter(param_str, {"ratio": float})
    return UnknownSentenceDropoutTrainer(**param)
  elif method == "normal":
    return UnknownNormalTrainer()
  else:
    raise ValueError("Unknown unknown_training method:", method)

