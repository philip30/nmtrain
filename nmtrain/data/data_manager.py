import nmtrain
import nmtrain.enumeration

class DataManager:

  def load_train(self, src, trg, src_voc, trg_voc, src_dev=None, trg_dev=None, batch_size=1):
    self.train_batches = load_parallel_data(src, trg, src_voc, trg_voc,
                                            mode=nmtrain.enumeration.DataMode.TRAIN,
                                            n_items=batch_size)
    if src_dev and trg_dev:
      self.dev_batches = load_parallel_data(src_dev, trg_dev, src_voc, trg_voc,
                                            mode=nmtrain.enumeration.DataMode.TEST,
                                            n_items=1)

  def load_test(self, src, trg, src_voc, trg_voc):
    self.test_batches = load_parallel_data(src, trg, src_voc, trg_voc,
                                           mode=nmtrain.enumeration.DataMode.TEST,
                                           n_items=1)
  # Training data arrange + shuffle
  def arrange(self, indexes):
    if indexes is not None:
      self.train_batches[0].arrange(indexes)
      self.train_batches[1].arrange(indexes)

  def shuffle(self):
    new_arrangement = self.train_batches[0].shuffle()
    self.train_batches[1].arrange(new_arrangement)
    return new_arrangement

  # Generators
  def train_data(self): return self.data(self.train_batches)
  def dev_data(self): return self.data(self.dev_batches)
  def test_data(self): return self.data(self.test_batches)

  def data(self, batches):
    for src, trg in zip(batches[0], batches[1]):
      yield src, trg

### Functions
def load_parallel_data(src, trg, src_vocab, trg_vocab, mode, n_items=1, cut_threshold=1):
  """ Load parallel data into batch managers """
  return (load_data(src, src_vocab, mode, n_items, cut_threshold),
          load_data(trg, trg_vocab, mode, n_items, cut_threshold))

def load_data(data, vocab, mode, n_items=1, cut_threshold=1):
  """ Transform single dataset into a form of batch manager """
  batch_manager = nmtrain.BatchManager()
  transformer = nmtrain.data.transformer.NMTDataTransformer(mode,
                                                            vocab=vocab,
                                                            unk_freq_threshold=cut_threshold)
  with open(data) as data_file:
    batch_manager.load(data_file,
                       n_items=n_items,
                       data_transformer=transformer,
                       sort=True)

  return batch_manager
