import collections
import chainer
import numpy
import functools

import nmtrain

class Lexicon(object):
  def __init__(self, src_voc, trg_voc, lexicon_alpha, lexicon_type, lexicon_file=None):
    self.alpha = lexicon_alpha
    self.type  = lexicon_type
    self.trg_size = len(trg_voc)
    self.unk_src_id = src_voc.unk_id()
 
    if lexicon_file is not None:
      self.data = lexicon_from_file(lexicon_file, src_voc, trg_voc)

  def init(self, src_data):
    src_size, batch_size = src_data.shape
    lexicon_matrix = numpy.zeros((batch_size, src_size, self.trg_size), dtype=numpy.float32)
    for i in range(batch_size):
      for j in range(src_size):
        lexicon_matrix[i][j] = self.dense_probability(src_data[j][i])

    return lexicon_matrix

  @functools.lru_cache(maxsize=512)
  def dense_probability(self, src_word):
    sparse_prob = self.data.lexicon.get(src_word, None)
    dense_prob = numpy.zeros(self.trg_size, dtype=numpy.float32)
    if sparse_prob:
      for trg_word, trg_prob in sparse_prob.data.items():
        dense_prob[trg_word] = trg_prob
    return dense_prob

def lexicon_from_file(lexicon_file, src_voc, trg_voc):
  lexicon_proto = nmtrain.dictionary_pb.Lexicon()
  lexicon_prob = lexicon_proto.lexicon
  with open(lexicon_file) as lex_fp:
    unparsed = 0
    for i, line in enumerate(lex_fp):
      try:
        trg, src, prob = line.rstrip().split()
      except:
        unparsed += 1
        continue
      trg = trg_voc.parse_word(trg)
      src = src_voc.parse_word(src)
      if trg != trg_voc.unk_id() and src != src_voc.unk_id():
        dict_prob = lexicon_prob[src].data
        dict_prob[trg] = dict_prob.get(trg, 0.0) + float(prob)
  if unparsed != 0:
    nmtrain.log.warning("There are %d/%d parsed lines from the lexicon" % (i-unparsed, i))
 
  # Making sure the probability is correct
  for src, p_trg_given_src in lexicon_prob.items():
    unk_prob = 1 - sum(p_trg_given_src.data.values())
    assert unk_prob >= -1e-6 and unk_prob <= 1 + 1e-6, "Strange unknown prob:%s" % (unk_prob)
    lexicon_prob[src].data[trg_voc.unk_id()] = unk_prob
  return lexicon_proto

def unk_replace_lexicon_from_file(lexicon_file):
  max_rep  = {}
  max_prob = {}
  with open(lexicon_file) as lex_fp:
    unparsed = 0
    for i, line in enumerate(lex_fp):
      try:
        trg, src, prob = line.strip().split()
      except:
        unparsed += 1
        continue
      prob = float(prob)
      if src not in max_prob or max_prob[src] < prob:
        max_prob[src] = prob
        max_rep[src] = trg
  if unparsed != 0:
    nmtrain.log.warning("There are %d/%d parsed lines from the lexicon" % (i-unparsed, i))
  return max_rep

