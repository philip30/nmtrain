#!/usr/bin/env python3

import math
import nmtrain.evals.bleu as bleu
import unittest

class TestBLEU(unittest.TestCase):
  def setUp(self):
    self.hyp = ["the taro met the hanako".split()]
    self.ref = ["taro met hanako".split()]

  def test_bleu_1gram(self):
    exp_bleu = 3.0 / 5.0
    act_bleu = bleu.calculate_bleu_corpus(self.hyp, self.ref, ngram=1).score
    self.assertEqual(act_bleu, exp_bleu)

  def test_bleu_4gram(self):
    exp_bleu = math.exp(math.log((3/5) * (2/5) * (1/4) * (1/3))/4)
    act_bleu = bleu.calculate_bleu_corpus(self.hyp, self.ref, ngram=4, smooth=1).score
    self.assertEqual(act_bleu, exp_bleu)

if __name__ == '__main__':
  unittest.main()
