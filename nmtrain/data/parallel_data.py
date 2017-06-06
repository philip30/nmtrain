import nmtrain
import math

# Represent Single Sentence
class SingleSentence(object):
  def __init__(self, line_number, original):
    self.line_number = line_number
    self.original    = original
    self.last_key    = "original"

  def annotate(self, key, value):
    setattr(self, key, value)
    self.last_key = key

  def content(self):
    return getattr(self, self.last_key)

  def __len__(self):
    return len(self.content())

  def __iter__(self):
    for word in self.content():
      yield word

  def __str__(self):
    return str(self.content())

# Represents Parallel Sentence
class ParallelSentence(object):
  def __init__(self, src_sent, trg_sent):
    self.src_sent = src_sent
    self.trg_sent = trg_sent

  def __len__(self):
    if self.trg_sent is not None:
      if self.src_sent is not None:
        return max(len(self.trg_sent), len(self.src_sent))
      else:
        return len(self.trg_sent)
    elif self.src_sent is not None:
      return len(self.src_sent)
    else:
      return 0

  def __hash__(self):
    src = len(self.src_sent) if self.src_sent is not None else -1
    trg = len(self.trg_sent) if self.trg_sent is not None else -1
    return hash((src, trg))

  def __iter__(self):
    yield self.src_sent
    yield self.trg_sent

  def __str__(self):
    return str(self.src_sent) + " ||| " + str(self.trg_sent)

# Generate a pair of sentence given parallel corpus
def data_generator(src_data, trg_data):
  if src_data is not None and trg_data is not None:
    for src, trg in zip(src_data, trg_data):
      yield ParallelSentence(src, trg)
  elif src_data is not None:
    for src in src_data:
      yield ParallelSentence(src, None)
  elif trg_data is not None:
    for trg in trg_data:
      yield ParallelSentence(None, trg)
  else:
    raise ValueError("Both of src_data and trg_data is None??")

# Represent ParallelCorpus
class ParallelData(object):
  def __init__(self, src, trg, batch_manager,
               wordid_converter=None, n_items=1,
               max_item=0,
               analyzer=None, filterer=None,
               sorter=None, bpe_codec=None):
    # The information about the location of its data
    self.src_path      = src
    self.trg_path      = trg
    self.batch_manager = batch_manager

    ### Begin Loading data
    preprocessor = nmtrain.data.preprocessor.NMTDataPreprocessor()
    def load_data(data, codec):
      corpus = []
      with open(data) as data_file:
        for line_number, line in enumerate(data_file):
          corpus.append(preprocessor(SingleSentence(line_number, line), codec))
      return corpus

    src_codec, trg_codec = (None, None) if bpe_codec is None else bpe_codec
    src_data, trg_data = None, None

    if src is not None and src:
      src_data = load_data(src, src_codec)
    if trg is not None and trg:
      trg_data = load_data(trg, trg_codec)
    # They need to be equal, otherwise they are not parallel data
    if src_data is not None and trg_data is not None:
      assert(len(src_data) == len(trg_data))

    ### Filter data
    data = data_generator(src_data, trg_data)
    if filterer is not None:
      data = filterer(data)
    else:
      data = list(data)

    ### Sort data
    if sorter is not None:
      data = sorter(data)

    ### Analyze data
    if analyzer is not None:
      analyzer(data)

    # Load the data with batch manager
    self.batch_manager.load(data, n_items=n_items, max_item=max_item, postprocessor=wordid_converter)

  def __iter__(self):
    for batch in self.batch_manager:
      yield batch
