import numpy

import nmtrain

class Batch(object):
  """ Class to represent batch """
  def __init__(self, batch_id, data):
    self.id   = batch_id
    self.data = data

  def __iter__(self):
    return iter(self.data)

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

class BatchManager(object):
  """ Class to manage batch reveal the indexes to public and retrieve
      the index by looking it in the map.
  """

  def __init__(self, strategy="sent"):
    # Hold the batch indexes
    self.batch_indexes    = []
    # Mapping from id -> batch
    self.batch_map  = {}
    self.strategy   = strategy

  # stream  : data stream
  # n_items : number of items in batch
  def load(self, stream, n_items=1, postprocessor=None):
    assert(n_items >= 1)

    partial_batch = lambda: None
    partial_batch.id      = 0
    partial_batch.data    = []
    partial_batch.length  = 0
    partial_batch.max_len = 0
    partial_batch.hash    = None

    def new_batch():
      batch = Batch(batch_id = partial_batch.id,
                    data     = partial_batch.data)
      if postprocessor is not None:
        postprocessor(batch)

      # Added the new batch
      self.batch_indexes.append(partial_batch.id)
      self.batch_map[partial_batch.id] = batch
      # Reset the state of patial_batch
      partial_batch.data = []
      partial_batch.id  += 1
      partial_batch.length = 0

    if self.strategy == "word":
      length_assess = lambda sent: len(sent)
    else:
      length_assess = lambda sent: 1

    # Creating batch
    for i, data in enumerate(stream):
      length = length_assess(data)
      if length + partial_batch.length > n_items and partial_batch.length != 0:
        new_batch()
      partial_batch.data.append(data)
      partial_batch.length += length

    if len(partial_batch.data) != 0:
      new_batch()

  def arrange(self, indexes):
    self.batch_indexes = list(indexes)

  def shuffle(self, random):
    random.shuffle(self.batch_indexes)
    return self.batch_indexes

  # Operators
  def __len__(self):
    return len(self.batch_indexes)

  def __getitem__(self, index):
    return self.batch_map[self.batch_indexes[index]]

  def __iter__(self):
    for index in self.batch_indexes:
      yield self.batch_map[index]

