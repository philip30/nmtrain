import numpy

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
  def __init__(self, dropout_ratio=0.2):
    self.ratio = dropout_ratio

  def dropout_word(self, batch):
    flag = numpy.random.rand(*batch.shape) >= self.ratio
    return batch * flag

  def __iter__(self):
    yield lambda batch: (self.dropout_word(batch.normal_data[0]), \
                         self.dropout_word(batch.normal_data[1]))

class UnknownSentenceDropoutTrainer(UnknownTrainer):
  def __init__(self, dropout_ratio=0.2):
    self.ratio = dropout_ratio

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
    param = util.parse_parameter(param_str, {"ratio": float})
    return UnknownWordDropoutTrainer(**param)
  elif method == "sentence_dropout":
    param = util.parse_parameter(param_str, {"ratio": float})
    return UnknownSentenceDropoutTrainer(**param)
  else:
    return UnknownNormalTrainer()

