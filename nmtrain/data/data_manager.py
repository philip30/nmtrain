import numpy

import nmtrain
import nmtrain.enumeration
import nmtrain.data.analyzer

def load_data(data, transformer):
  corpus = []
  with open(data) as data_file:
    for line in data_file:
      corpus.append(transformer.transform(line))
  return corpus

def transform_corpus(corpus, analyzer, transformer, vocab):
  for line in corpus:
    analyzer.analyze(line)
  analyzer.finish_analysis()
  transformer.transform_corpus(corpus, analyzer, vocab)

class ParallelData:
  def __init__(self, src, trg, src_voc, trg_voc, mode,
               n_items=1, cut_threshold=0, src_max_vocab=1e8, trg_max_vocab=1e8,
               max_sent_length=100, sort=True):
    self.src_batch_manager = nmtrain.BatchManager()
    self.trg_batch_manager = nmtrain.BatchManager()
    self.src_analyzer = nmtrain.data.analyzer.StandardAnalyzer(max_vocab_size=src_max_vocab, unk_freq_threshold=cut_threshold)
    self.trg_analyzer = nmtrain.data.analyzer.StandardAnalyzer(max_vocab_size=trg_max_vocab, unk_freq_threshold=cut_threshold)

    # Data Transformer
    transformer = nmtrain.data.transformer.NMTDataTransformer(data_type=mode)

    # Begin Loading data
    src_data = load_data(src, transformer)
    if trg is not None:
      trg_data = load_data(trg, transformer)

      # They need to be equal, otherwise they are not parallel data
      assert(len(src_data) == len(trg_data))
   
    # Max sent length cut + sort
    if mode == nmtrain.enumeration.DataMode.TRAIN:
      src_crp = []
      trg_crp = []
      if sort:
        data = sorted(zip(src_data, trg_data), key= lambda line: (len(line[1]), len(line[0])))
      else:
        data = zip(src_data, trg_data)
      for src_line, trg_line in data:
        if len(src_line) <= max_sent_length and len(trg_line) <= max_sent_length:
          src_crp.append(src_voc.add_sentence(src_line))
          trg_crp.append(trg_voc.add_sentence(trg_line))
      src_data = src_crp
      trg_data = trg_crp
    else:
      src_data = list(map(lambda sent: src_voc.parse_sentence(sent), src_data))
      if trg is not None:
        trg_data = list(map(lambda sent: trg_voc.parse_sentence(sent), trg_data))
   
    # Analyzing corpus
    transform_corpus(src_data, self.src_analyzer, transformer, src_voc)
    if trg is not None:
      transform_corpus(trg_data, self.trg_analyzer, transformer, trg_voc)

    # Post processes
    src_pp = lambda batch: transformer.transform_batch(batch, src_voc)
    trg_pp = lambda batch: transformer.transform_batch(batch, trg_voc)

    # Load the data with batch manager
    self.src_batch_manager.load(src_data, n_items=n_items, post_process=src_pp)
    if trg is not None:
      self.trg_batch_manager.load(trg_data, n_items=n_items, post_process=trg_pp)

  def __iter__(self):
    if len(self.trg_batch_manager) != 0:
      for src, trg in zip(self.src_batch_manager, self.trg_batch_manager):
        yield src, trg
    else:
      for src in self.src_batch_manager:
        yield src, None

  def src(self):
    for src in self.src_batch_manager:
      yield src

  def trg(self):
    for trg in self.trg_batch_manager:
      yield trg

class DataManager:
  def load_train(self, src, trg, src_voc, trg_voc,
                 src_dev=None, trg_dev=None,
                 src_test=None, trg_test=None,
                 batch_size=1, unk_cut=0, src_max_vocab=1e6,
                 max_sent_length=100,
                 trg_max_vocab=1e6):
    # Loading Training Data
    self.train_data = ParallelData(src, trg, src_voc, trg_voc,
                                   mode=nmtrain.enumeration.DataMode.TRAIN,
                                   n_items=batch_size,
                                   cut_threshold=unk_cut,
                                   src_max_vocab=src_max_vocab,
                                   trg_max_vocab=trg_max_vocab,
                                   max_sent_length=max_sent_length,
                                   sort=True)
    self.src_dev = src_dev
    self.trg_dev = trg_dev
    self.src_test = src_test
    self.trg_test = trg_test

    if src_dev and trg_dev:
      self.dev_data = ParallelData(src_dev, trg_dev, src_voc, trg_voc,
                                   mode=nmtrain.enumeration.DataMode.TEST,
                                   n_items=1, sort=False)
    if src_test and trg_test:
      self.test_data = ParallelData(src_test, trg_test, src_voc, trg_voc,
                                    mode=nmtrain.enumeration.DataMode.TEST,
                                    n_items=1, sort=False)

  def load_test(self, src, src_voc, trg_voc, ref=None):
    self.test_data = ParallelData(src, trg=ref, src_voc=src_voc, trg_voc=trg_voc,
                                  mode=nmtrain.enumeration.DataMode.TEST,
                                  n_items=1, sort=False)

  # Training data arrange + shuffle
  def arrange(self, indexes):
    if indexes is not None:
      self.train_data.src_batch_manager.arrange(indexes)
      self.train_data.trg_batch_manager.arrange(indexes)

  def shuffle(self):
    new_arrangement = self.train_data.src_batch_manager.shuffle()
    self.train_data.trg_batch_manager.arrange(new_arrangement)
    return new_arrangement

  def has_dev_data(self):
    return hasattr(self, "dev_data")

  def has_test_data(self):
    return hasattr(self, "test_data")

  def total_trg_words(self):
    return self.train_data.trg_analyzer.total_count

