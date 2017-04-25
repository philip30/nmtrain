import nmtrain
import numpy
import unittest

class TestBatchManager(unittest.TestCase):
  def setUp(self):
    numpy.random.seed(17)
    self.manager = nmtrain.data.BatchManager()

  def test_load_batch_data(self):
    """ Test load batch data.
        batch {
          d0, d1
        }
        batch {
          d2, d3
        }
        batch {
          d4
        }
    """
    data = ["1st", "2nd", "3rd", "4th", "5th"]
    self.manager.load(data, n_items = 2)
    batches = self.manager
    self.assertEqual(len(batches[0].data), 2)
    self.assertEqual(len(batches[1].data), 2)
    self.assertEqual(len(batches[2].data), 1)

    self.assertEqual(batches[0].data[0], "1st")
    self.assertEqual(batches[0].data[1], "2nd")
    self.assertEqual(batches[1].data[0], "3rd")
    self.assertEqual(batches[1].data[1], "4th")
    self.assertEqual(batches[2].data[0], "5th")

  def test_shuffle(self):
    data = ["1st", "2nd", "3rd", "4th", "5th"]
    self.manager.load(data, n_items = 1)
    self.manager.shuffle(numpy.random.RandomState(seed=17))
    batches = self.manager

    self.assertEqual(batches[0].data[0], "5th")
    self.assertEqual(batches[1].data[0], "1st")
    self.assertEqual(batches[2].data[0], "3rd")
    self.assertEqual(batches[3].data[0], "4th")
    self.assertEqual(batches[4].data[0], "2nd")

  def test_iter(self):
    data = ["1st", "2nd", "3rd", "4th", "5th"]
    expected = ["5th", "1st", "3rd", "4th", "2nd"]
    self.manager.load(data, n_items = 1)
    self.manager.shuffle(numpy.random.RandomState(seed=17))
    for i, batch in enumerate(self.manager):
      self.assertEqual(batch.data[0], expected[i])

  def test_arrange(self):
    data = ["1st", "2nd", "3rd", "4th", "5th"]
    expected = ["5th", "1st", "2nd", "4th", "3rd"]
    self.manager.load(data, n_items = 1)
    self.manager.arrange([4, 0, 1, 3, 2])
    for i, batch in enumerate(self.manager):
      self.assertEqual(batch.data[0], expected[i])

  def test_even_data(self):
    data = ["1st", "2nd", "3rd", "4th"]
    self.manager.load(data, n_items = 2)
    self.assertEqual(len(self.manager), 2)

  def test_transform_integer(self):
    class TransformInteger:
      def __call__(self, partial_batch):
        def transform(data):
          if data == "1st":
            return 1
          elif data == "2nd":
            return 2
          elif data == "3rd":
            return 3
          elif data == "4th":
            return 4
          else:
            return 5
        for i, data in enumerate(partial_batch.data):
          partial_batch.data[i] = transform(data)
    data = ["1st", "2nd", "3rd", "4th", "5th"]
    expected = [1, 2, 3, 4, 5]
    self.manager.load(data, n_items=1, postprocessor=TransformInteger())
    for i, batch in enumerate(self.manager):
      self.assertEqual(batch.data[0], expected[i])

  def test_word_data(self):
    data = ["1", "12", "123", "12345","123456", "12345678"]
    expected = [["1", "12", "123"], ["12345"], ["123456"], ["12345678"]]
    manager = nmtrain.data.BatchManager(strategy="word")
    manager.load(data, n_items=6)
    for batch, expected in zip(manager, expected):
      self.assertEqual(batch.data, expected)

if __name__ == "__main__":
  unittest.main()
