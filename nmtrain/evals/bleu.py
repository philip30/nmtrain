import math
import numpy
import itertools
import functools
import sys

from collections import defaultdict, deque

def calculate_bleu_corpus(hypothesis, reference, ngram=4, smooth=0):
  hyp_stats = []
  ref_stats = []
  for hyp, ref in itertools.zip_longest(hypothesis, reference):
    # DEBUG BEGIN
    #print("Comparing:\n\t%s\n\t%s" % (str(hyp), str(ref)), file=sys.stderr)
    # DEBUG END
    # Update the dictionary with the stats of n_grams
    hyp_stats.append(ngram_stats(sentence = hyp, gram = ngram))
    ref_stats.append(ngram_stats(sentence = ref, gram = ngram))

  # Calculating BLEU score
  return BLEU(hyp_stats, ref_stats, order=ngram, smooth=smooth)

def calculate_bleu_sentence_fast(hypothesis, reference, ngram=4, smooth=1):
  hyp_stats = numeric_ngram_stats(hypothesis, gram=ngram)
  ref_stats = numeric_ngram_stats(reference, gram=ngram)
  log_precision = 0
  for i, (hyp_stat, ref_stat) in enumerate(zip(hyp_stats, ref_stats)):
    tp, denom = 0, 0
    for word, word_count in hyp_stat.items():
      tp += min(word_count, ref_stat.get(word, 0))
    denom = sum(hyp_stat.values())
    log_precision += math.log((tp + smooth) / (denom + smooth))
  len_hyp = len(hypothesis)
  len_ref = len(reference)
  if len(reference) != 0 and len_hyp < len_ref:
    log_bp = 1 - (len_ref / len_hyp)
  else:
    log_bp = 0
  assert(log_precision <= 0)
  score =  math.exp(log_precision / ngram + log_bp)
  return score

class BLEU(object):
  def __init__(self, hyp_ngrams, ref_ngrams, order, smooth=0):
    self.precisions      = []
    self.score           = 0
    self.brevity_penalty = 1
    self.hyp_length      = 0
    self.ref_length      = 0
    self.order           = order
    self.stats           = []

    # The length is the sum of all unigram tokens
    self.hyp_length = hyp_length = sum(sum(stat[0].values()) for stat in hyp_ngrams)
    self.ref_length = ref_length = sum(sum(stat[0].values()) for stat in ref_ngrams)
    # Calculate BLEU for every n-gram
    for i in range(order):
      smooth_val = 0 if i == 0 else smooth
      true_positive, denom = 0, 0
      for hyp_stat, ref_stat in zip(hyp_ngrams, ref_ngrams):
        for word, word_count in hyp_stat[i].items():
          if word in ref_stat[i]:
            true_positive += min(word_count, ref_stat[i][word])
        denom += sum(hyp_stat[i].values())
      self.stats.append((true_positive, denom))
      if denom + smooth_val == 0:
        precision = 0
      else:
        precision = (true_positive + smooth_val) / (denom + smooth_val)

      self.precisions.append(precision)
    # Calculate brevity penalty
    if hyp_length < ref_length:
      if hyp_length == 0:
        self.brevity_penalty = 0
      else:
        self.brevity_penalty= math.exp(1 - (ref_length / hyp_length))

    # Calculate score
    if any(x < 1e-6 for x in self.precisions):
      self.score = 0.0
    else:
      self.score = math.exp(sum(map(math.log, self.precisions)) /
        len(self.precisions)) * self.brevity_penalty

  def value(self):
    return self.score

  def __lt__(self, other):
    return self.score < other.score

  def __gt__(self, other):
    return self.score > other.score

  def __eq__(self, other):
    return self.score == other.score

  def __str__(self):
    return str(self.score * 100) + " BP=%.4f" % (self.brevity_penalty) + \
           " (%s)" % (", ".join("%d/%d" % (tp, ln) for tp, ln in self.stats))

def ngram_stats(sentence, gram):
  output_dict = defaultdict(lambda: defaultdict(int))
  for i in range(len(sentence)):
    for j in range(1, gram+1):
      if i+j < len(sentence) + 1:
        output_dict[j-1][" ".join(map(str, sentence[i:i+j]))] += 1
  return output_dict

@functools.lru_cache(maxsize=512)
def numeric_ngram_stats(sentence, gram):
  stats = [defaultdict(int) for _ in range(gram)]
  for ngram in range(gram):
    window = deque()
    for word in sentence:
      window.append(word)
      if len(window) > ngram+1:
        window.popleft()
      if len(window) == ngram+1:
        stats[ngram][hash(tuple(window))] += 1
  return stats

