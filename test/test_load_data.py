import nmtrain
import unittest

import test.util as util

class TestLoadData(unittest.TestCase):
  def setUp(self):
    nmtrain.log.silence(True)
    self.manager = nmtrain.data.DataManager()
    self.src_voc = nmtrain.Vocabulary(add_eos=True, add_unk=True)
    self.trg_voc = nmtrain.Vocabulary(add_eos=True, add_unk=True)
    self.args = util.basic_train_args(src="load.src", trg="load.trg")

  def tearDown(self):
    nmtrain.log.silence(False)

  def test_load_train(self):
    train_data = self.manager.load_train(src = self.args.src,\
                                         trg = self.args.trg,
                                         src_voc = self.src_voc,
                                         trg_voc = self.trg_voc,
                                         batch_size = 2)

if __name__ == "__main__":
  unittest.main()
