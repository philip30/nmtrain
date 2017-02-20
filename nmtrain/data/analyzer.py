from collections import defaultdict

# TODO(philip30): Add test to this class
class StandardAnalyzer(object):
  def __init__(self, max_vocab_size=1e8, unk_freq_threshold=0):
    self.word_count = defaultdict(int)
    self.total_count = 0
    self.max_vocab_size = max_vocab_size
    self.unk_freq_threshold = unk_freq_threshold
    self.max_sent_length = 0

  def analyze(self, sentence):
    for word_id in sentence:
      self.word_count[word_id] += 1
      self.total_count += 1
    self.max_sent_length = max(self.max_sent_length, len(sentence))

  def finish_analysis(self):
    keys = sorted(self.word_count.keys(), key=lambda key: self.word_count[key], reverse=True)
    if len(keys) > self.max_vocab_size:
      self.in_vocab = set(keys[:self.max_vocab_size])
    else:
      self.in_vocab =  set(keys)

  def is_rare_word(self, word_id):
    return (self.word_count[word_id] <= self.unk_freq_threshold) or (word_id not in self.in_vocab)
