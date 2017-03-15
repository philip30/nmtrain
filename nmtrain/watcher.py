import math
import numpy
import time

import nmtrain
import nmtrain.evals as eval
import nmtrain.log as log

class TrainingWatcher(object):
  def __init__(self, state, src_vocab, trg_vocab, early_stop_num):
    self.state = state
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
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
    # To measure number of trained sentence
    self.state.trained_sentence = 0
    # Verbose
    log.info("Start Epoch %d" % (self.state.finished_epoch + 1))

  def batch_begin(self):
    self.batch_time = time.time()

  def batch_update(self, id, loss=0, batch_size=1, col_size=1):
    ppl = math.exp(float(loss))
    self.epoch_loss += loss
    self.epoch_update_counter += 1
    self.state.trained_sentence += batch_size
    log.info("[%d] Sentence trained: %d, Batch(PPL=%f, size=(%d,%d), wps=%d, id=%d)" % (self.state.finished_epoch + 1,
                                                                                        self.state.trained_sentence,
                                                                                        ppl,
                                                                                        batch_size, col_size,
                                                                                        abs((batch_size * col_size) / (time.time() - self.batch_time)),
                                                                                        id))

  def end_epoch(self, new_data_arrangement):
    self.state.finished_epoch += 1
    self.state.batch_indexes = new_data_arrangement
    self.state.time_spent.append(time.time() - self.time)
    self.state.perplexities.append(math.exp(self.epoch_loss / self.epoch_update_counter))
    log.info("Epoch %d finished! ppl=%.4f, time=%.4f mins" % (self.state.finished_epoch,
                                                              self.state.ppl(),
                                                              self.state.last_time() / 60))
  # DEV SET
  # Sentence-wise prediction
  def start_prediction(self):
    pass

  def end_prediction(self, loss, prediction=None):
    self.dev_loss += float(loss)
    self.dev_loss_ctr += 1
    if prediction is not None:
      self.predictions.append(prediction)
    # During Training ignore attention vector.
    # This might change in the future

  # Corpus-wise evalution
  def begin_evaluation(self):
    self.dev_loss = 0
    self.dev_loss_ctr = 0
    self.predictions = []
    log.info("Begin Evaluation...")

  def end_evaluation(self, dev_data, trg_vocab):
    if len(self.predictions) != 0:
      self.state.bleu_scores.append(calculate_bleu(self.predictions,
                                                   trg_dev,
                                                   trg_vocab))
    # Adding previous dev perplexities if available
    if len(self.state.dev_perplexities) > 0:
      prev_dev_ppl = self.state.dev_ppl()
    else:
      prev_dev_ppl = None

    # Update the dev perplexities
    self.state.dev_perplexities.append(math.exp(self.dev_loss / self.dev_loss_ctr))

    # Generate one line report
    dev_ppl = self.state.dev_ppl()
    if prev_dev_ppl is None:
      dev_ppl_report = "DEV_PPL=%.3f" % dev_ppl
    else:
      dev_ppl_report = "DEV_PPL=(%.3f -> %.3f)" % (prev_dev_ppl, dev_ppl)

    # Appending BLEU scores information if generated
    if len(self.state.bleu_scores) != 0:
      dev_ppl_report += ", BLEU=%s" % self.state.bleu()

    # Reporting
    log.info("End Evaluation: %s" % (dev_ppl_report))

  def end_of_sentence(self, word):
    return word == self.trg_vocab.eos_id()

  def should_save(self):
    if len(self.state.dev_perplexities) > 0:
      best_ppl_index = int(numpy.argmin(self.state.dev_perplexities))
      return (best_ppl_index + 1) == len(self.state.dev_perplexities)
    else:
      return True

  def should_early_stop(self):
    if len(self.state.dev_perplexities) > 0:
      best_ppl_index = int(numpy.argmin(self.state.dev_perplexities))
      return abs(len(self.state.dev_perplexities) - best_ppl_index - 1) > self.early_stop
    return False

class TestWatcher(object):
  def __init__(self, state, src_vocab, trg_vocab, output_stream=None):
    self.state     = state
    self.src_vocab = src_vocab
    self.trg_vocab = trg_vocab
    self.output_stream = output_stream

  def begin_evaluation(self):
    log.info("Decoding Started")
    self.time           = time.time()
    self.predictions    = []
    self.test_loss      = 0
    self.test_loss_ctr  = 0
    self.attentions     = []

  def end_evaluation(self, test_data, trg_vocab):
    log.info("Decoding Finished, starting evaluation if reference is provided.")
    self.state.time_spent.append(time.time() - self.time)
    ref = test_data.trg_path
    if ref is not None:
      self.state.bleu_scores.append(calculate_bleu(self.predictions, ref, trg_vocab))
      self.state.perplexities.append(math.exp(self.test_loss / self.test_loss_ctr))
    # Creating evaluation string
    eval_string = "Time=%.2f mins" % (self.state.time() / 60)
    if ref is not None:
      eval_string += " " + ("BLEU=%s, test_ppl=%f" % (str(self.state.bleu()), self.state.ppl()))

    log.info("Evaluation Finished!", eval_string)

  def end_of_sentence(self, word):
    return word == self.trg_vocab.eos_id()

  def start_prediction(self):
    pass

  def end_prediction(self, loss, prediction, probabilities, attention):
    if prediction is not None:
      self.predictions.append(prediction)
    if loss is not None:
      self.test_loss     += float(loss)
      self.test_loss_ctr += 1

    # Attention is SRC X TRG
    if attention is not None:
      self.attentions.append(attention)

    if self.output_stream is not None:
      print(prediction, file=self.output_stream)
      self.output_stream.flush()

# Calculate BLEU Score
def calculate_bleu(predictions, ref, trg_vocab):
  def src_corpus():
    for hyp in predictions:
      yield hyp.split()
  def trg_corpus():
    # TODO(philip30): If you modify transformer, also consider modifying this.
    with open(ref) as ref_file:
      for line in ref_file:
        yield line.strip().split()
  return eval.bleu.calculate_bleu_corpus(src_corpus(), trg_corpus(), verbose=False)

