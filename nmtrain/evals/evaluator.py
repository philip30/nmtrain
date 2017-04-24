import nmtrain

from nmtrain.evals import bleu

class Evaluator(object):
  def __init__(self, config):
    self.config = config

  def reset(self):
    self.prediction = []

  def assess_sentence_level(self, prediction_output, batch):
    score = {}
    if prediction_output is None:
      return score
    self.prediction.append(prediction_output.prediction.split())
    return score

  def assess_corpus_level(self, data):
    def ref_generator():
      for ref in data.batch_manager:
        yield ref[0].trg_sent.tokenized
    score = {}
    if len(self.prediction) != 0:
      for bleu_config in self.config.bleu:
        key = bleu_config.annotation
        if key == "":
          key = "bleu-" + str(bleu_config.ngram) + ("-s(%d)" % bleu_config.smooth)
        value = bleu.calculate_bleu_corpus(hypothesis = self.prediction,
                                           reference  = ref_generator(),
                                           ngram      = bleu_config.ngram,
                                           smooth     = bleu_config.smooth)
        score[key] = value.value()

    return score
