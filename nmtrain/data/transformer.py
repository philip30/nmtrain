import numpy

import nmtrain
from . import analyzer

class IdentityTransformer(object):
  def transform(self, data, load_mode=nmtrain.enum.DataMode.TRAIN):
    return data

  def transform_corpus(self, corpus_data):
    return

class NMTDataTransformer(object):
  def __init__(self, data_type=nmtrain.enum.DataMode.TRAIN,
               vocab=nmtrain.vocabulary.Vocabulary(),
               data_analyzer=analyzer.StandardAnalyzer(),
               unk_freq_threshold=1):
    self.vocab = vocab
    self.data_type = data_type
    self.data_analyzer = data_analyzer
    self.unk_freq_threshold = unk_freq_threshold

  def transform(self, data):
    """ Transform the input string to list of word id.
        If it is training, add it. Otherwise just try to parse it.
    """
    sentence = []
    for word in data.strip().split():
      if self.data_type == nmtrain.enum.DataMode.TRAIN:
        sentence.append(self.vocab.add_word(word))
      else:
        sentence.append(self.vocab.parse_word(word))
    return sentence
    
  def transform_corpus(self, corpus):
    """ Called after all data is being transformed. 
        Transform further according to the corpus analysis.

        The transformation includes:
          1. Adding stuff "{*}" to the end of the element of the batch so they have equal length
          2. Adding EOS for every element in batch
          3. Transform it into imutable numpy array.
    """
    for batch_id, batch_data in corpus.items():
      max_length = max([len(batch) for batch in batch_data])
      for sentence_id, sentence in enumerate(batch_data):
        # First if it is train model the unknown word for freq < threshold
        if self.data_type == nmtrain.enum.DataMode.TRAIN:
          for i, word_id in enumerate(sentence):
            if self.data_analyzer.word_count[self.vocab.word(word_id)] < self.unk_freq_threshold:
              sentence[i] = self.vocab.set_rare_word(word_id)
        # Stuff it so they have a square batch
        for _ in range(max_length - len(sentence)):
          sentence.append(self.vocab.stuff_id())
        # Add the end of word at the end
        sentence.append(self.vocab.eos_id())
        # Overwrite the existing
        batch_data[sentence_id] = sentence
      batch_data = numpy.array(batch_data, dtype=numpy.int)
      corpus[batch_id] = batch_data
    return corpus
