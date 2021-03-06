import nmtrain
import math
import time

import nmtrain.log as log

class Watcher(object):
  def __init__(self, state):
    self.state = state.data

  def begin_train_epoch(self):
    log.info("Start Epoch %d" % (self.state.train_state.finished_epoch + 1))
    self.epoch_update = self.state.train_updates.add()

  def begin_dev_epoch(self):
    self.epoch_update = self.state.dev_updates.add()

  def begin_test_epoch(self):
    self.epoch_update = self.state.test_updates.add()

  def end_train_epoch(self):
    self.end_epoch("train")

  def end_dev_epoch(self, score):
    self.end_epoch("dev", score)

  def end_test_epoch(self, score):
    self.end_epoch("test", score)

  def begin_batch(self):
    self.batch_update = self.epoch_update.batch_updates.add()
    self.batch_update.time = time.time()

  def end_batch(self, loss, src_shape, trg_shape, batch_id):
    self.record_updates(loss, batch_id, trg_shape)
    wps = float(self.batch_update.trained_words) / self.batch_update.time

    if loss >= 0:
      loss_str = "ppl=%10.3f" % self.batch_update.score["ppl"]
    else:
      loss_str = "loss=%.10f" % self.batch_update.score["loss"]

    # Logging as needed
    log.info("[%d] Processed: %8d, %s, size=%3d,(%3d,%3d), wps=%.3f" \
             % (self.state.train_state.finished_epoch+1,
                self.epoch_update.trained_sentence,
                loss_str,
                src_shape[1], src_shape[0],
                trg_shape[0], wps))

    # Remove reference
    self.batch_update = None

  def update_epoch(self, time_taken, trained_word, trained_sent):
    self.epoch_update.time += time_taken
    self.epoch_update.trained_words += trained_word
    self.epoch_update.trained_sentence += trained_sent

  def record_updates(self, loss, batch_id, trg_shape, score = None):
    time_taken = time.time() - self.batch_update.time
    trained_word = trg_shape[0] * trg_shape[1]
    trained_sent = trg_shape[1]
    # Batch Updates
    self.batch_update.batchid = batch_id
    self.batch_update.time = time_taken
    self.batch_update.trained_words = trained_word
    self.batch_update.trained_sentence = trained_sent
    # Epoch Updates
    self.update_epoch(time_taken, trained_word, trained_sent)

    if loss is not None:
      ppl = math.exp(float(loss))
      self.batch_update.score["loss"] = float(loss)
      self.batch_update.score["ppl"] = ppl
      self.epoch_update.score["loss"] += float(loss)
      self.epoch_update.score["ppl"] += ppl

    if score is not None:
      for eval_key, eval_value in score.items():
        self.epoch_update.score[eval_key] = eval_value

  def end_epoch(self, prefix, score=None):
    if self.epoch_update.time != 0:
      wps = self.epoch_update.trained_words / self.epoch_update.time
      wps_str = "%5.3f" % wps
    else:
      wps_str = "-"
    if self.epoch_update.batch_updates:
      self.epoch_update.score["loss"] /= len(self.epoch_update.batch_updates)
      self.epoch_update.score["ppl"]  /= len(self.epoch_update.batch_updates)
    if score is not None:
      for eval_key, eval_value in score.items():
        self.epoch_update.score[eval_key] = eval_value
    # Logging as needed
    if self.epoch_update.score["loss"] != 0 and self.epoch_update.score["ppl"] != 0:
      if self.epoch_update.score["loss"] <= 0:
        ppl_str = "loss=%3.10f, " % self.epoch_update.score["loss"]
      else:
        ppl_str = "ppl=%3.3f, " % self.epoch_update.score["ppl"]
    else:
      ppl_str = ""
    if score is not None:
      score_str = ", ".join(key + "=" + ("%4.3f" % value) for key, value in sorted(score.items()))
      score_str += ", "
    else:
      score_str = ""
    log.info("%5s: [%d] %s%stime=%5.3f mins, wps=%5.3f" \
              % (prefix.upper(), self.state.train_state.finished_epoch + 1,
                 ppl_str, score_str,
                 self.epoch_update.time / 60, wps))
    # Remove reference
    self.epoch_update = None

  def new_epoch(self):
    self.state.train_state.finished_epoch += 1

