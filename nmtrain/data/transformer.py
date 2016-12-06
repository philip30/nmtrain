import numpy

import nmtrain

class IdentityTransformer(object):
  def transform(self, data, load_mode=nmtrain.enumeration.DataMode.TRAIN):
    return data

  def transform_corpus(self, corpus_data):
    return

  def transform_batch(self, batch):
    return

class NMTDataTransformer(object):
  def __init__(self, data_type):
    self.data_type = data_type

  def transform(self, data):
    return data.strip().split()

  def transform_corpus(self, corpus, analyzer, vocab):
    """ Called after all data is being transformed.
        Transform further according to the corpus analysis.
    """
    # This is just applied to the training corpus
    if self.data_type != nmtrain.enumeration.DataMode.TRAIN:
      return
    # First map the unknown words
    for i, sentence in enumerate(corpus):
      # First if it is train model the unknown word for freq < threshold
      for j, word_id in enumerate(sentence):
        if analyzer.is_rare_word(word_id):
          sentence[j] = vocab.set_rare_word(word_id)
    # Remapping the unknown words for training data
    # Making the unknown words to appear at the last of the ids
    unk_map = vocab.remap_unknown()
    # Remap the whole corpus
    for i, sentence in enumerate(corpus):
      for j, word_id in enumerate(sentence):
        if word_id in unk_map:
          new_word_id, new_word = unk_map[word_id]
          sentence[j] = new_word_id
        else:
          raise Exception("Unmapped word id:", word_id)
    # Return the new corpus (the old one has already changed also)
    return corpus

  def transform_batch(self, batch, vocab):
    # stuffing
    max_len = max(len(line) for line in batch.data)
    for i, line in enumerate(batch.data):
      stuffs = [vocab.stuff_id()] * (max_len - len(line))
      line.extend(stuffs)
      line.append(vocab.eos_id())
    batch.data = numpy.array(batch.data, dtype=numpy.int32).transpose()
