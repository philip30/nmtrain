import numpy

import nmtrain
import nmtrain.data.transformer as transformer

class Batch(object):
  """ Class to represent batch """
  def __init__(self, batch_id, data):
    self.id   = batch_id
    self.data = data

class BatchManager(object):
  """ Class to manage batch reveal the indexes to public and retrieve
      the index by looking it in the map.
  """
  
  def __init__(self):
    # Hold the batch indexes
    self.batch_indexes    = []
    # Mapping from id -> batch
    self.batch_map  = {}

  # stream  : data stream
  # n_items : number of items in batch
  def load(self, stream, n_items=1, 
           data_transformer = transformer.IdentityTransformer()):
    assert(n_items >= 1)
   
    partial_batch = lambda: None
    partial_batch.id   = 0
    partial_batch.data = []

    def new_batch():
      batch = Batch(batch_id = partial_batch.id,
          data = partial_batch.data)
      # Added the new batch
      self.batch_indexes.append(partial_batch.id)
      self.batch_map[partial_batch.id] = batch
      # Reset the state of patial_batch
      partial_batch.data = []
      partial_batch.id  += 1
   
    # Load data from stream
    for i, data in enumerate(stream):
      partial_batch.data.append(data_transformer.transform(data))
      if len(partial_batch.data) == n_items:
        new_batch()
    
    if len(partial_batch.data) != 0:
      new_batch()

    data_transformer.transform_corpus(self.batch_map)
  
  def arrange(self, indexes):
    self.batch_indexes = indexes

  def shuffle(self):
    numpy.random.shuffle(self.batch_indexes)
  
  # Operators
  def __len__(self):
    return len(self.batch_indexes)

  def __getitem__(self, index):
    return self.batch_map[self.batch_indexes[index]]

  def __iter__(self):
    for index in self.batch_indexes:
      yield self.batch_map[index]
