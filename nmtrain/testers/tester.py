import nmtrain
import chainer

from nmtrain.evals import evaluator
from nmtrain.testers.post_processor import TestPostProcessor

DEV  = 1
TEST = 2

class Tester(object):
  def __init__(self, watcher, classifier, model, outputer, config):
    self.watcher = watcher
    self.classifier = classifier
    self.eos_id = model.trg_vocab.eos_id()
    self.evaluator = evaluator.Evaluator(config.evaluation)
    self.post_processor = TestPostProcessor(model.src_vocab, model.trg_vocab, config.post_process)
    self.config = config

  def __call__(self, *args, **kwargs):
    with chainer.no_backprop_mode():
      with chainer.using_config('train', False):
        self.test(*args, **kwargs)

  def test(self, model, data, mode, outputer):
    nmtrain.log.info("Begin Testing")
    self.epoch(mode, "begin")
    self.evaluator.reset()
    ### Variables
    eval_ppl  = self.config.evaluation.eval_ppl
    predict   = len(self.config.evaluation.bleu) != 0
    ### Test Loop
    for batch in data:
      src_sent, trg_sent = batch.normal_data
      self.watcher.begin_batch()
      # Evaluating PPL with beam=1
      if eval_ppl and trg_sent is not None:
        loss = self.classifier.train(model, src_sent, trg_sent, self.eos_id)
      else:
        loss = None
      # Doing prediction if other score is wished
      if predict:
        predict_output = self.classifier.predict(model, src_sent,
                                                 eos_id       = self.eos_id,
                                                 word_penalty = self.config.word_penalty,
                                                 gen_limit    = self.config.generation_limit,
                                                 beam         = self.config.beam)
        self.post_processor(predict_output, batch)
        outputer(batch.data[0].src_sent.bpe_merge, predict_output, id=batch.id+1)
      else:
        predict_output = None
      loss = loss.data if loss is not None else 0
      self.watcher.record_updates(loss     = loss,
                                  score    = self.evaluator.assess_sentence_level(predict_output, batch),
                                  batch_id = batch.id,
                                  trg_shape = trg_sent.shape)
    self.epoch(mode, "end", self.evaluator.assess_corpus_level(data))

  def epoch(self, mode, text, *args):
    if mode == DEV:
      method = "dev"
    elif mode == TEST:
      method = "test"
    else:
      raise ValueError("Unknown mode:", mode)
    reflect_method = "%s_%s_epoch" % (text, method)
    getattr(self.watcher, reflect_method)(*args)
