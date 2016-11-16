from collections import defaultdict

# TODO(philip30): Add test to this class
class StandardAnalyzer(object):
  def __init__(self):
    self.word_count = defaultdict(int)
    self.total_count = 0

  def add_word_count(self, word):
    self.word_count[word] += 1
    self.total_count += 1
