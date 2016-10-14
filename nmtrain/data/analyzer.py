from collections import defaultdict

# TODO(philip30): Add test to this class
class StandardAnalyzer(object):
  def __init__(self):
    self.word_count = defaultdict(int)

  def add_word_count(self, word):
    self.word_count[word] += 1
