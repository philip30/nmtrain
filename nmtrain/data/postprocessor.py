import numpy

class WordIdConverter(object):
  def __init__(self, src_vocab, trg_vocab, analyzer=None, include_rare=False):
    self.src_vocab       = src_vocab
    self.trg_vocab       = trg_vocab
    self.analyzer        = analyzer
    self.include_rare    = include_rare
    self.max_trg_corpus  = (0, 0)

    if analyzer is not None:
      self.src_vocab.set_check_rare(analyzer.is_src_rare)
      self.trg_vocab.set_check_rare(analyzer.is_trg_rare)

  def __call__(self, finished_batch):
    src_max_len = -1
    trg_max_len = -1
    for par_sent in finished_batch.data:
      src = par_sent.src_sent
      trg = par_sent.trg_sent
      if src is not None:
        src_max_len = max(len(src), src_max_len)
      if trg is not None:
        trg_max_len = max(len(trg), trg_max_len)
        if trg_max_len > self.max_trg_corpus[0]:
          self.max_trg_corpus = trg_max_len, finished_batch.id

    src_data = []
    trg_data = []

    # Function to parse data to word id and stuff it
    def process_data(data, max_len, sentence, vocab):
      is_parse = self.analyzer is None or vocab.is_frozen()
      if sentence is not None:
        stuffs = [vocab.eos_id()] * (max_len - len(sentence))
        if is_parse:
          wids   = vocab.parse_sentence(sentence)
        else:
          wids   = vocab.add_sentence(sentence, include_rare=self.include_rare)
        wids.extend(stuffs)
        wids.append(vocab.eos_id())
        data.append(wids)

    for i, par_sent in enumerate(finished_batch.data):
      process_data(src_data, src_max_len, par_sent.src_sent, self.src_vocab)
      process_data(trg_data, trg_max_len, par_sent.trg_sent, self.trg_vocab)

    def convert(data):
      if len(data) == 0:
        return None
      else:
        return numpy.array(data, dtype=numpy.int32).transpose()

    # This is the data that is normally used in NMT. It is a ready to use batch
    finished_batch.normal_data = convert(src_data), convert(trg_data)

    # This data is a normal_data with unknown.
    # This data should be equal to normal_data
    if self.include_rare:
      def process_batch(batch, vocab):
        if batch is None:
          return None
        ret = numpy.empty_like(batch)
        ret[:] = batch
        for i, batch_word in enumerate(ret):
          for j, word_id in enumerate(batch_word):
            if vocab.check_rare(vocab.word(word_id)) and \
                not vocab.check_special_id(word_id):
              ret[i][j] = vocab.unk_id()
        return ret

      src_batch, trg_batch = finished_batch.normal_data
      finished_batch.unk_data = process_batch(src_batch, self.src_vocab), \
                                process_batch(trg_batch, self.trg_vocab)
    return

