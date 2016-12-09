import numpy

import nmtrain

class Tester(object):
  def __init__(self, data, classifier, watcher, trg_vocab, predict=True, eval_ppl=True):
    self.data      = data
    self.predict   = predict
    self.eval_ppl  = eval_ppl
    self.watcher   = watcher
    self.trg_vocab = trg_vocab
    self.classifier = classifier

  def test(self, model, word_penalty, gen_limit, beam_size):
    xp = nmtrain.environment.array_module()
    self.watcher.begin_evaluation()
    for src_sent, trg_sent in self.data.test_data:
      if xp != numpy:
        src_data = xp.array(src_sent.data, dtype=numpy.int32)
        trg_data = xp.array(trg_sent.data, dtype=numpy.int32) if trg_sent is not None else None
      else:
        src_data = src_sent.data
        trg_data = trg_sent.data if trg_sent is not None else None
      self.watcher.start_prediction()
      if self.eval_ppl and trg_sent is not None:
        loss = self.classifier.eval(model, src_data, trg_data)
      else:
        loss = None
      if self.predict:
        predict_output = self.classifier.predict(model, src_data,
                                                 eos_id       = self.trg_vocab.eos_id(),
                                                 trg_data     = trg_data,
                                                 word_penalty = word_penalty,
                                                 gen_limit    = gen_limit,
                                                 beam         = beam_size)
      else:
        predict_output = lambda: None
        predict_output.prediction = None
        predict_output.attention  = None
        predict_output.probabilities = None
      self.watcher.end_prediction(loss          = loss,
                                  prediction    = predict_output.prediction,
                                  attention     = predict_output.attention,
                                  probabilities = predict_output.probabilities)
    self.watcher.end_evaluation(self.data.src_test,
                                self.data.trg_test,
                                self.trg_vocab)
