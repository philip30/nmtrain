import numpy
import unittest

import nmtrain
import nmtrain.batch
import nmtrain.environment
import nmtrain.data.transformer as transformer

class TestNMTTransformer(unittest.TestCase):
  def setUp(self):
    # Vocabulary
    vocab = nmtrain.Vocabulary(True, True, True)

    # Analyzer
    data_analyzer = nmtrain.data.analyzer.StandardAnalyzer()
    self.train_corpus = ["this is a test .", "this is philip arthur"]
    self.train_transformer = transformer.NMTDataTransformer(nmtrain.enumeration.DataMode.TRAIN,
                                                            vocab=vocab,
                                                            data_analyzer=data_analyzer,
                                                            unk_freq_threshold=1)

  def test_transform_line(self):
    word_id = self.train_transformer.transform("this is philip arthur")
    self.assertEqual(word_id, [3, 4, 5, 6])

  def test_transform_batch(self):
    batch = []
    batch.append(self.train_transformer.transform("this is philip arthur"))
    batch.append(self.train_transformer.transform("this is a test ."))
    corpus = {0 : nmtrain.batch.Batch(0, batch)}
    self.train_transformer.transform_corpus(corpus)
    expected_batch = numpy.array([[3, 4, 5, 6, 2, 1], [3, 4, 7, 8, 9, 1]])
    numpy.testing.assert_array_equal(corpus[0].data, expected_batch)

class TestNMTTestTransformer(unittest.TestCase):
  def setUp(self):
    vocab = nmtrain.Vocabulary(True, True, True)
    train_data = ["this is philip arthur", "this is a test"]
    for line in train_data:
      for word in line.split():
        vocab.add_word(word)
    self.test_transformer = transformer.NMTDataTransformer(nmtrain.enumeration.DataMode.TEST,
                                                           vocab = vocab)

  def test_transform_line(self):
    test_transformed = self.test_transformer.transform("this is test data")
    self.assertEqual([3, 4, 8, 0], test_transformed)

class TestRareWords(unittest.TestCase):
  def setUp(self):
    # Vocabulary
    vocab = nmtrain.Vocabulary(True, True, True)

    # Analyzer
    data_analyzer = nmtrain.data.analyzer.StandardAnalyzer()
    self.train_corpus = ["this is a test .", "this is philip arthur"]
    self.train_transformer = transformer.NMTDataTransformer(nmtrain.enumeration.DataMode.TRAIN,
                                                            vocab=vocab,
                                                            data_analyzer=data_analyzer,
                                                            unk_freq_threshold=2)

  def test_transform_batch(self):
    batch = []
    batch.append(self.train_transformer.transform("this is philip arthur"))
    batch.append(self.train_transformer.transform("this is a test ."))
    corpus = {0 : nmtrain.batch.Batch(0, batch)}
    self.train_transformer.transform_corpus(corpus)
    expected_batch = numpy.array([[3, 4, 0, 0, 2, 1], [3, 4, 0, 0, 0, 1]])
    numpy.testing.assert_array_equal(corpus[0].data, expected_batch)

if __name__ == "__main__":
  unittest.main()
