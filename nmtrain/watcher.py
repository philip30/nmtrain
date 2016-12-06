import math
import numpy
import time

import nmtrain
import nmtrain.evals as eval
import nmtrain.log as log

class TrainingWatcher(object):
  def __init__(self, state, src_vocab, trg_vocab, total_trg_words, early_stop_num):
    self.state = state
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.total_trg_words = total_trg_words
    self.early_stop = early_stop_num

  # TRAIN SET
  def begin_epoch(self):
    # Number of training sentences
    self.trained = 0
    # To measure ppl
    self.epoch_loss = 0
    self.epoch_update_counter = 0
    # To measure time
    self.time = time.time()
    # Verbose
    log.info("Start Epoch %d" % (self.state.finished_epoch + 1))

  def batch_update(self, loss=0, batch_size=1, col_size=1):
    ppl = math.exp(float(loss))
    self.epoch_loss += loss
    self.epoch_update_counter += 1
    self.trained += batch_size
    log.info("[%d] Sentence trained: %d, Batch_PPL=%f, column size=%d" % (self.state.finished_epoch + 1, self.trained, ppl, col_size))

  def end_epoch(self, new_data_arrangement):
    self.state.finished_epoch += 1
    self.state.batch_indexes = new_data_arrangement
    self.state.time_spent.append(time.time() - self.time)
    self.state.perplexities.append(math.exp(self.epoch_loss / self.epoch_update_counter))
    self.state.wps_time.append(self.total_trg_words / self.state.last_time())
    log.info("Epoch %d finished! PPL=%f, time=%f mins, wps=%f" % (self.state.finished_epoch,
                                                                  self.state.ppl(),
                                                                  self.state.last_time() / 60,
                                                                  self.state.wps()))
  # DEV SET
  # Sentence-wise prediction
  def start_prediction(self):
    pass

  def end_prediction(self, loss, prediction, probabilities, attention):
    self.dev_loss += float(loss)
    self.dev_loss_ctr += 1
    self.predictions.append(prediction)
    # During Training ignore attention vector.
    # This might change in the future

  # Corpus-wise evalution
  def begin_evaluation(self):
    self.dev_loss = 0
    self.dev_loss_ctr = 0
    self.predictions = []
    log.info("Begin Evaluation...")

  def end_evaluation(self, src_dev, trg_dev, trg_vocab):
    self.state.bleu_scores.append(calculate_bleu(self.predictions, trg_dev, trg_vocab))
    self.state.dev_perplexities.append(math.exp(self.dev_loss/ self.dev_loss_ctr))

    # Generate one line report
    dev_ppl = self.state.dev_ppl()
    dev_ppl_report = "DEV_PPL=%f" % dev_ppl if dev_ppl < 1e6 else "DEV_PPL=TOO_BIG"
    log.info("End Evaluation: %s, BLEU=%s" % (dev_ppl_report,
                                              self.state.bleu()))

  def end_of_sentence(self, word):
    return word == self.trg_vocab.eos_id()

  def should_save(self):
    if len(self.state.bleu_scores) > 0:
      highest_bleu_index = int(numpy.argmax(self.state.bleu_scores))
      return (highest_bleu_index + 1) == len(self.state.bleu_scores)
    else:
      return True

  def should_early_stop(self):
    if len(self.state.bleu_scores) > 0:
      highest_bleu_index = int(numpy.argmax(self.state.bleu_scores))
      return abs(len(self.state.bleu_scores) - highest_bleu_index - 1) > self.early_stop
    return False

class TestWatcher(object):
  def __init__(self, state, src_vocab, trg_vocab, output_stream=None):
    self.state     = state
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.output_stream = output_stream

  def begin_evaluation(self):
    log.info("Decoding Started")
    self.time = time.time()
    self.predictions = []
    self.test_loss = 0
    self.test_loss_ctr = 0
    self.attentions = []

  def end_evaluation(self, src, ref, trg_vocab):
    log.info("Decoding Finished, starting evaluation if reference is provided.")
    self.state.time_spent.append(time.time() - self.time)
    self.state.wps_time.append(sum(len(prediction) for prediction in self.predictions) / self.state.last_time())
    if ref is not None:
      self.state.bleu_scores.append(calculate_bleu(self.predictions, ref, trg_vocab))
      self.state.perplexities.append(math.exp(self.test_loss / self.test_loss_ctr))
    # Creating evaluation string
    eval_string = "Time=%.2f mins, WPS=%f" % (self.state.time() / 60, self.state.wps())
    if ref is not None:
      eval_string += " " + ("BLEU=%s, PPL=%f" % (str(self.state.bleu()), self.state.ppl()))

    log.info("Evaluation Finished!", eval_string)

  def end_of_sentence(self, word):
    return word == self.trg_vocab.eos_id()

  def start_prediction(self):
    pass

  def end_prediction(self, loss, prediction, probabilities, attention):
    self.predictions.append(prediction)
    self.test_loss += float(loss)
    self.test_loss_ctr += 1

    # Attention is SRC X TRG
    if attention is not None:
      self.attentions.append(attention)

    if self.output_stream is not None:
      print(self.trg_vocab.sentence(prediction), file=self.output_stream)
      self.output_stream.flush()

# Calculate BLEU Score
def calculate_bleu(predictions, ref, trg_vocab):
  def src_corpus():
    for hyp in predictions:
      yield trg_vocab.sentence(hyp).split()
  def trg_corpus():
    # TODO(philip30): If you modify transformer, also consider modifying this.
    with open(ref) as ref_file:
      for line in ref_file:
        yield line.strip().split()
  return eval.bleu.calculate_bleu_corpus(src_corpus(), trg_corpus(), verbose=False)

