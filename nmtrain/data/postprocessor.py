import numpy

class WordIdConverter(object):
  def __init__(self, src_vocab, trg_vocab, analyzer=None):
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.analyzer  = analyzer

    if analyzer is not None:
      self.src_vocab.set_check_rare(analyzer.is_src_rare)
      self.trg_vocab.set_check_rare(analyzer.is_trg_rare)

  def __call__(self, finished_batch):
    src_max_len = -1
    trg_max_len = -1
    for par_sent in finished_batch.data:
      src = par_sent.src_sent
      trg = par_sent.trg_sent
      src_max_len = max(len(src), src_max_len)
      if trg is not None:
        trg_max_len = max(len(trg), trg_max_len)

    src_data = []
    trg_data = []

    def process_data(data, max_len, sentence, vocab):
      if sentence is not None:
        stuffs = [vocab.eos_id()] * (max_len - len(sentence))
        if self.analyzer is None:
          wids   = vocab.parse_sentence(sentence)
        else:
          wids   = vocab.add_sentence(sentence)
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
    finished_batch.final_data = convert(src_data), convert(trg_data)

