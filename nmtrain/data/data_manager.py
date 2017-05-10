import numpy
import copy

import nmtrain
import nmtrain.data.sorter
import nmtrain.data.analyzer
import nmtrain.data.preprocessor
import nmtrain.data.postprocessor
import nmtrain.third_party.bpe

from nmtrain.data.parallel_data import ParallelData

""" This class is a manager of data. It holds all the corpora that are loaded from the file with various setting. """
class DataManager(object):
  def load_train(self, corpus, data_config, nmtrain_model):
    src_vocab = nmtrain_model.src_vocab
    trg_vocab = nmtrain_model.trg_vocab
    bpe_codec = nmtrain_model.bpe_codec
    # Sorter to sort the batch
    sorter = nmtrain.data.sorter.from_string(data_config.sort_method)
    # Analyzer to get some statistics of the corpus
    analyzer = nmtrain.data.analyzer.ParallelCountAnalyzer(data_config.src_max_vocab,
                                                           data_config.trg_max_vocab,
                                                           data_config.unk_cut)
    # Whether to include rare in the normal batch or not. 
    # It depends on the unknown training method (if != normal)
    include_rare = data_config.unknown_training.method != "normal"
    # Filterer is used to delete some elements in the corpus
    filterer = nmtrain.data.preprocessor.FilterSentence(data_config.max_sent_length)
    # These converters will convert the string to ID and fill that in to the vocabulary
    # Or it will simply use the vocabulary
    self.train_converter = nmtrain.data.postprocessor.WordIdConverter(src_vocab, trg_vocab, analyzer, include_rare)
    self.test_converter  = nmtrain.data.postprocessor.WordIdConverter(src_vocab, trg_vocab)
    # Retain some properties
    self.analyzer = analyzer

    # Loading Training Data
    self.train_data = ParallelData(src              = corpus.train_data.source,
                                   trg              = corpus.train_data.target,
                                   batch_manager    = nmtrain.data.BatchManager(data_config.batch_strategy),
                                   n_items          = data_config.batch,
                                   analyzer         = analyzer,
                                   filterer         = filterer,
                                   sorter           = sorter,
                                   bpe_codec        = bpe_codec,
                                   wordid_converter = self.train_converter)

    # Loading Dev Data if available
    if corpus.dev_data.source and corpus.dev_data.target:
      self.dev_data = ParallelData(src              = corpus.dev_data.source,
                                   trg              = corpus.dev_data.target,
                                   n_items          = 1,
                                   batch_manager    = nmtrain.data.BatchManager("sent"),
                                   bpe_codec        = bpe_codec,
                                   wordid_converter = self.test_converter)

    # Loading Test Data if available
    if corpus.test_data.source and corpus.test_data.target:
      self.test_data = ParallelData(src              = corpus.test_data.source,
                                    trg              = corpus.test_data.target,
                                    batch_manager    = nmtrain.data.BatchManager("sent"),
                                    n_items          = 1,
                                    bpe_codec        = bpe_codec,
                                    wordid_converter = self.test_converter)

    # random state for shuffling batch
    self.random = numpy.random.RandomState(seed = nmtrain_model.config.seed)
    self.random_ctr = -1

    # return the training data
    return self.train_data

  def load_test(self, data, nmtrain_model):
    src_vocab = nmtrain_model.src_vocab
    trg_vocab = nmtrain_model.trg_vocab
    bpe_codec = nmtrain_model.bpe_codec
    test_converter  = nmtrain.data.postprocessor.WordIdConverter(src_vocab, trg_vocab)
    self.test_data = ParallelData(src              = data.source,
                                  trg              = data.target,
                                  batch_manager    = nmtrain.data.BatchManager("sent"),
                                  n_items          = 1,
                                  bpe_codec        = bpe_codec,
                                  wordid_converter = test_converter)
    return self.test_data

  # Training data arrange + shuffle
  def arrange(self, epoch, force_put_max=False):
    while self.random_ctr < epoch:
      self.random.shuffle(self.train_data.batch_manager.batch_indexes)
      self.random_ctr += 1
      # Put the longest target batch here
      if epoch == 0 or force_put_max:
        nmtrain.log.info("Switching the longest trg batch first")
        max_stat = self.train_converter.max_trg_corpus
        current = self.train_data.batch_manager.batch_indexes
        max_index = current.index(max_stat[1])
        current[0], current[max_index] = current[max_index], current[0]


  @property
  def has_dev_data(self):
    return hasattr(self, "dev_data")

  @property
  def has_test_data(self):
    return hasattr(self, "test_data")

