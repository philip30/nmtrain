import chainer
import shutil
import numpy
import unittest

import nmtrain
import nmtrain.classifiers
import nmtrain.trainers
import nmtrain.serializer
import nmtrain.log as log
import test.util as util

class TestSerializer(unittest.TestCase):
  def setUp(self):
    log.silence()
    util.init_env()
    args = util.basic_train_args()
    # Make the model
    self.trainer = nmtrain.trainers.MaximumLikelihoodTrainer(args)
    self.trainer.train(nmtrain.classifiers.RNN_NMT())
    self.model_file = args.model_out
    log.silence(False)

  def tearDown(self):
    shutil.rmtree(self.model_file)

  def test_serialize(self):
    act = nmtrain.NmtrainModel(util.basic_test_args(self.model_file))
    exp = self.trainer.nmtrain_model
    self.assertEqual(act.__class__, exp.__class__)
    self.check_vocab_equal(act.src_vocab, exp.src_vocab)
    self.check_vocab_equal(act.trg_vocab, exp.trg_vocab)
    self.check_training_state_equal(act.training_state, exp.training_state)
    self.check_chainer_model_equal(act.chainer_model, exp.chainer_model)

  def check_vocab_equal(self, act, exp):
    self.assertEqual(act.__class__, exp.__class__)
    self.assertEqual(len(act), len(exp))
    self.assertEqual(act.rare_words, exp.rare_words)
    self.assertEqual(act.word_to_id, exp.word_to_id)
    self.assertEqual(act.id_to_word, exp.id_to_word)

  def check_training_state_equal(self, act, exp):
    self.assertEqual(act.finished_epoch, exp.finished_epoch)
    self.assertEqual(act.perplexities, exp.perplexities)
    self.assertEqual(act.dev_perplexities, exp.dev_perplexities)
    self.assertEqual(act.bleu_scores, exp.bleu_scores)
    self.assertEqual(act.time_spent, exp.time_spent)
    self.assertEqual(act.wps_time, exp.wps_time)
    self.assertEqual(act.batch_indexes, exp.batch_indexes)

  def check_chainer_model_equal(self, act, exp):
    self.assertEqual(act.__class__, exp.__class__)
    self.assertEqual(len(act._params), len(exp._params))
    # Check for parameters
    for act_param, exp_param in zip(act._params, exp._params):
      act_param = getattr(act, act_param)
      exp_param = getattr(exp, exp_param)
      numpy.testing.assert_array_equal(act_param.data, exp_param.data)

    # Recursively checking for children
    if isinstance(act, chainer.ChainList):
      self.assertEqual(len(act), len(exp))
      for act_link, exp_link in zip(act, exp):
        self.check_chainer_model_equal(act_link, exp_link)
    else:
      if not hasattr(act, "_children"):
        return
      for act_child, exp_child in zip(act._children, exp._children):
        act_child = getattr(act, act_child)
        exp_child = getattr(exp, exp_child)

        self.check_chainer_model_equal(act_child, exp_child)

if __name__ == "__main__":
  unittest.main()
