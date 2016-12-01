import math

from collections import defaultdict

def calculate_bleu_corpus (hypothesis, reference, ngram=4, smooth=0):
  ref_count = 0

  hyp_ngrams = defaultdict(lambda: defaultdict(int))
  ref_ngrams = defaultdict(lambda: defaultdict(int))
  for hyp, ref in zip(hypothesis, reference):
    # Update the dictionary with the stats of n_grams
    n_gram_stats(sentence = hyp, gram = ngram, output_dict = hyp_ngrams)
    n_gram_stats(sentence = ref, gram = ngram, output_dict = ref_ngrams)
    ref_count += 1

  assert(len(hypothesis) == ref_count)

  # Calculating BLEU score
  return BLEU(hyp_ngrams, ref_ngrams, order=ngram, smooth=smooth)

class BLEU(object):
  def __init__(self, hyp_ngrams, ref_ngrams, order, smooth=0):
    self.precisions      = []
    self.score           = 0
    self.brevity_penalty = 1
    self.hyp_length      = 0
    self.ref_length      = 0
    self.order           = order

    # The length is the sum of all unigram tokens
    self.hyp_length = hyp_length = sum(hyp_ngrams[0].values())
    self.ref_length = ref_length = sum(ref_ngrams[0].values())
    # Calculate BLEU for every n-gram
    for i in range(order):
      smooth_val = 0 if i == 0 else smooth
      hyp_ngram = hyp_ngrams[i]
      ref_ngram = ref_ngrams[i]
      true_positive = sum([min(word_count, ref_ngrams[i][word]) for word, word_count in hyp_ngrams[i].items()])
      denom = sum(hyp_ngram.values())

      self.precisions.append((true_positive + smooth_val)/(denom + smooth_val))
    # Calculate brevity penalty
    if hyp_length < ref_length:
      self.brevity_penalty= math.exp(1 - (ref_length / hyp_length))

    # Calculate score
    if any(x < 1e-6 for x in self.precisions):
      self.score = 0.0
    else:
      self.score = math.exp(sum(map(math.log, self.precisions)) /
        len(self.precisions)) * self.brevity_penalty

  def value(self):
    return self.score

  def __str__(self):
    return str(self.score * 100) + " BP=%f" % (self.brevity_penalty)

def n_gram_stats(sentence, gram, output_dict):
  for i in range(len(sentence)):
    for j in range(1, gram+1):
      if i+j < len(sentence) + 1:
        output_dict[j-1][" ".join(map(str, sentence[i:i+j]))] += 1
  return output_dict
