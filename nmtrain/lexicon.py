import collections
import chainer
import numpy
import functools

import nmtrain

class Lexicon(object):
  def __init__(self, lexicon_file, src_voc, trg_voc, lexicon_alpha, lexicon_type):
    self.alpha = lexicon_alpha
    self.type  = lexicon_type
    self.lexicon = lexicon_from_file(lexicon_file, src_voc, trg_voc)
    self.trg_size = len(trg_voc)
    self.unk_src_id = src_voc.unk_id()

  def init(self, src_data):
    src_size, batch_size = src_data.shape
    lexicon_matrix = numpy.zeros((batch_size, src_size, self.trg_size), dtype=numpy.float32)
    for i in range(batch_size):
      for j in range(src_size):
        lexicon_matrix[i][j] = self.dense_probability(src_data[j][i])

    # Convert to gpu / cpu array
    xp = nmtrain.environment.array_module()
    if xp != numpy:
      lexicon_matrix = xp.array(lexicon_matrix, dtype=numpy.float32)
    self.lexicon_matrix = nmtrain.environment.Variable(lexicon_matrix)

  @functools.lru_cache(maxsize=512)
  def dense_probability(self, src_word):
    sparse_prob = self.lexicon[src_word]
    dense_prob = numpy.zeros(self.trg_size, dtype=numpy.float32)
    for trg_word, trg_prob in sparse_prob.items():
      dense_prob[trg_word] = trg_prob
    dense_prob[self.unk_src_id] += (1 - sum(dense_prob))
    return dense_prob

  def p_lex(self):
    return self.lexicon_matrix

def lexicon_from_file(lexicon_file, src_voc, trg_voc):
  lexicon_prob = collections.defaultdict(lambda: collections.defaultdict(int))
  with open(lexicon_file) as lex_fp:
    for line in lex_fp:
      try:
        trg, src, prob = line.strip().split()
      except:
        raise ValueError("Failed to parse line for lexicon:", line)
      trg = trg_voc.parse_word(trg)
      src = src_voc.parse_word(src)
      lexicon_prob[src][trg] += float(prob)
  # Making sure the probability is correct
  for src, p_trg_given_src in lexicon_prob.items():
    assert abs(sum(p_trg_given_src.values()) - 1) < 1e-6, "p(trg|src='%s') does not sum to 1" % src_voc.word(src)
  return lexicon_prob

