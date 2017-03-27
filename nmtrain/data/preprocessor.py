class NMTDataPreprocessor(object):
  def __call__(self, data, codec):
    data.annotate("tokenized", self.tokenize(data))
    if codec is not None:
      data.annotate("encoded", self.bpe_encode(data, codec))
    return data.content()

  def tokenize(self, data):
    return data.original.strip().split()

  def bpe_encode(self, data, codec):
    return codec.segment(data.tokenized)

class FilterSentence(object):
  def __init__(self, max_sent_length=-1):
    self.max_sent_length = max_sent_length

  def __call__(self, data):
    if self.max_sent_length == -1:
      return list(data)

    ret = []
    for data_point in data:
      for single_data in data_point:
        if len(single_data) > self.max_sent_length:
          break
      else:
        ret.append(data_point)
    return ret
