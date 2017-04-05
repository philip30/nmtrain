import nmtrain
import unittest

def add_words(vocab):
  vocab.add_word("this")
  vocab.add_word("is")
  vocab.add_word("a")
  vocab.add_word("test")
  return vocab

class TestVocabulary(unittest.TestCase):
  def setUp(self):
    self.vocab = add_words(nmtrain.Vocabulary())

  def test_add_word(self):
    for word_id, word in enumerate(["this", "is", "a", "test"]):
      self.assertEqual(self.vocab[word], word_id)

  def test_get_sentence(self):
    sentence = self.vocab.sentence([0, 1, 2, 3])
    self.assertEqual(sentence, "this is a test")

  def test_get_word(self):
    self.assertEqual(self.vocab.word(0), "this")

class TestVocabularyPlus(unittest.TestCase):
  def setUp(self):
    self.vocab = add_words(nmtrain.Vocabulary(add_eos=True, add_unk=True))

  def test_unk_exists(self):
    self.assertEqual(self.vocab.unk_id(), 0)

  def test_eos_exists(self):
    self.assertEqual(self.vocab.eos_id(), 1)

  def test_get_sentence(self):
    word_ids = [2, 3, 4, 5] # 0, 1  are reserved for eos, unk, stuff
    sentence = self.vocab.sentence(word_ids)
    self.assertEqual(sentence, "this is a test")

  def test_eos_in_middle(self):
    word_ids = [2, 3, self.vocab.eos_id(), 1]
    sentence = self.vocab.sentence(word_ids)
    self.assertEqual(sentence, "this is")

  def test_parse_word(self):
    self.assertEqual(self.vocab.parse_word("this"), 2)

  def test_parse_word_unk(self):
    self.assertEqual(self.vocab.parse_word("philip"), 0)

  def test_parse_sentence(self):
    sentence = "this is philip arthur ."
    self.assertEqual(self.vocab.parse_sentence(sentence.split()), [2, 3, 0, 0, 0])

if __name__ == "__main__":
  unittest.main()
