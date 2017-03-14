import functools
from collections import defaultdict

class ParallelCountAnalyzer(object):
  def __init__(self, src_max_vocab=-1, trg_max_vocab=-1, unk_freq_threshold=0):
    self.src_analyzer       = CountAnalyzer()
    self.trg_analyzer       = CountAnalyzer()
    self.src_max_vocab      = src_max_vocab
    self.trg_max_vocab      = trg_max_vocab
    self.unk_freq_threshold = unk_freq_threshold

  def __call__(self, data):
    self.src_analyzer(src for src, trg in data)
    self.trg_analyzer(trg for src, trg in data)

  @functools.lru_cache(maxsize=4096)
  def is_src_rare(self, src_word):
    return self.src_analyzer.is_rare_word(src_word, self.src_max_vocab, self.unk_freq_threshold)

  @functools.lru_cache(maxsize=4096)
  def is_trg_rare(self, trg_word):
    return self.trg_analyzer.is_rare_word(trg_word, self.trg_max_vocab, self.unk_freq_threshold)

  @property
  def src_max_length(self):
    return self.src_analyzer.max_sent_length

  @property
  def trg_max_lenth(self):
    return self.trg_analyzer.max_sent_length

class CountAnalyzer(object):
  def __init__(self):
    self.word_count      = defaultdict(int)
    self.word_rank       = defaultdict(int)
    self.total_count     = 0
    self.max_sent_length = 0

  def __call__(self, data):
    # Analyze the corpus
    for sentence in data:
      for word in sentence:
        self.word_count[word] += 1
        self.total_count      += 1
      self.max_sent_length = max(self.max_sent_length, len(sentence))
    # Rank the corpus
    for rank, (word, _) in enumerate(sorted(self.word_count.items(),
                                            key=lambda x: x[1],
                                            reverse=True)):
      self.word_rank[word] = rank
    return

  def is_rare_word(self, word, max_vocab, unk_freq_threshold):
    rare = False
    if max_vocab >= 0:
      rare = self.word_rank[word] >= max_vocab
    if unk_freq_threshold >= 0:
      rare = rare or self.word_count[word] <= unk_freq_threshold
    return rare

