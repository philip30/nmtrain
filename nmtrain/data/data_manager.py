import numpy

import nmtrain
import nmtrain.data.sorter
import nmtrain.data.analyzer
import nmtrain.data.preprocessor
import nmtrain.data.postprocessor

from nmtrain.data.parallel_data import ParallelData

""" This class is a manager of data. It holds all the corpora that are loaded from the file with various setting. """
class DataManager(object):
  def load_train(self, src, trg, src_voc, trg_voc,
                 src_dev         = None,
                 trg_dev         = None,
                 src_test        = None,
                 trg_test        = None,
                 batch_size      = 1,
                 unk_cut         = 0,
                 src_max_vocab   = -1,
                 trg_max_vocab   = -1,
                 max_sent_length = -1,
                 sort_method     = "lentrg",
                 batch_strategy  = "sent",
                 bpe_codec       = None):
    sorter = nmtrain.data.sorter.from_string(sort_method)
    analyzer = nmtrain.data.analyzer.ParallelCountAnalyzer(src_max_vocab, trg_max_vocab, unk_cut)
    filterer = nmtrain.data.preprocessor.FilterSentence(max_sent_length)
    train_converter = nmtrain.data.postprocessor.WordIdConverter(src_voc, trg_voc, analyzer)
    test_converter  = nmtrain.data.postprocessor.WordIdConverter(src_voc, trg_voc)

    # Loading Training Data
    self.train_data = ParallelData(src              = src,
                                   trg              = trg,
                                   batch_manager    = nmtrain.data.BatchManager(batch_strategy),
                                   mode             = nmtrain.enumeration.DataMode.TRAIN,
                                   n_items          = batch_size,
                                   analyzer         = analyzer,
                                   filterer         = filterer,
                                   sorter           = sorter,
                                   bpe_codec        = bpe_codec,
                                   wordid_converter = train_converter)

    # Loading Dev Data if available
    if src_dev and trg_dev:
      self.dev_data = ParallelData(src              = src_dev,
                                   trg              = trg_dev,
                                   mode             = nmtrain.enumeration.DataMode.TEST,
                                   n_items          = 1,
                                   batch_manager    = nmtrain.data.BatchManager("sent"),
                                   bpe_codec        = bpe_codec,
                                   wordid_converter = test_converter)

    # Loading Test Data if available
    if src_test and trg_test:
      self.test_data = ParallelData(src              = src_test,
                                    trg              = trg_test,
                                    batch_manager    = nmtrain.data.BatchManager("sent"),
                                    mode             = nmtrain.enumeration.DataMode.TEST,
                                    n_items          = 1,
                                    bpe_codec        = bpe_codec,
                                    wordid_converter = test_converter)
    # return the training data
    return self.train_data

  def load_test(self, src, src_voc, trg_voc, ref=None, bpe_codec=None):
    test_converter  = nmtrain.data.postprocessor.WordIdConverter(src_voc, trg_voc)
    self.test_data = ParallelData(src              = src,
                                  trg              = ref,
                                  batch_manager    = nmtrain.data.BatchManager("sent"),
                                  mode             = nmtrain.enumeration.DataMode.TEST,
                                  n_items          = 1,
                                  bpe_codec        = bpe_codec,
                                  wordid_converter = test_converter)

  # Training data arrange + shuffle
  def arrange(self, indexes):
    if indexes is not None:
      self.train_data.batch_manager.arrange(indexes)

  def shuffle(self):
    return self.train_data.batch_manager.shuffle()

  @property
  def has_dev_data(self):
    return hasattr(self, "dev_data")

  @property
  def has_test_data(self):
    return hasattr(self, "test_data")

