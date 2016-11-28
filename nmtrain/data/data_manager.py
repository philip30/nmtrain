import nmtrain
import nmtrain.enumeration

class DataManager:

  def load_train(self, src, trg, src_voc, trg_voc,
                 src_dev=None, trg_dev=None,
                 src_test=None, trg_test=None, batch_size=1, unk_cut=0):
    self.train_batches = load_parallel_data(src, trg, src_voc, trg_voc,
                                            mode=nmtrain.enumeration.DataMode.TRAIN,
                                            n_items=batch_size,
                                            cut_threshold=unk_cut)
    if src_dev and trg_dev:
      self.dev_batches = load_parallel_data(src_dev, trg_dev, src_voc, trg_voc,
                                            mode=nmtrain.enumeration.DataMode.TEST,
                                            n_items=1)
    if src_test and trg_test:
      self.test_batches = load_parallel_data(src_test, trg_test, src_voc, trg_voc,
                                             mode=nmtrain.enumeration.DataMode.TEST,
                                             n_items=1)

  def load_test(self, src, src_voc, ref=None, trg_voc=None):
    if ref is not None:
      self.test_batches = load_parallel_data(src, ref, src_voc, trg_voc,
                                             mode=nmtrain.enumeration.DataMode.TEST,
                                             n_items=1, sort=False)
    else:
      self.test_batches = (load_data(src, src_voc,
                                     mode=nmtrain.enumeration.DataMode.TEST,
                                     n_items=1, sort=False), None)

  # Training data arrange + shuffle
  def arrange(self, indexes):
    if indexes is not None:
      self.train_batches[0].arrange(indexes)
      self.train_batches[1].arrange(indexes)

  def shuffle(self):
    new_arrangement = self.train_batches[0].shuffle()
    self.train_batches[1].arrange(new_arrangement)
    return new_arrangement

  def has_dev_data(self):
    return hasattr(self, "dev_batches")

  def has_test_data(self):
    return hasattr(self, "test_batches")

  def total_trg_words(self):
    return self.train_batches[1].analyzer.total_count

  # Generators
  def train_data(self): return self.data(self.train_batches)
  def dev_data(self): return self.data(self.dev_batches)
  def test_data(self): return self.data(self.test_batches)

  def data(self, batches):
    if batches[1] is None:
      for src in batches[0]:
        yield src, None
    else:
      for src, trg in zip(batches[0], batches[1]):
        yield src, trg

### Functions
def load_parallel_data(src, trg, src_vocab, trg_vocab, mode, n_items=1, cut_threshold=1, sort=True):
  """ Load parallel data into batch managers """
  return (load_data(src, src_vocab, mode, n_items, cut_threshold, sort=sort),
          load_data(trg, trg_vocab, mode, n_items, cut_threshold, sort=sort))

def load_data(data, vocab, mode, n_items=1, cut_threshold=1, sort=True):
  """ Transform single dataset into a form of batch manager """
  batch_manager = nmtrain.BatchManager()
  transformer = nmtrain.data.transformer.NMTDataTransformer(mode,
                                                            vocab=vocab,
                                                            unk_freq_threshold=cut_threshold,
                                                            data_analyzer=batch_manager.analyzer)

  with open(data) as data_file:
    batch_manager.load(data_file,
                       n_items=n_items,
                       data_transformer=transformer,
                       sort=sort)

  return batch_manager
