import math
import time

import nmtrain
import nmtrain.log as log

class TrainingWatcher(object):
  def __init__(self, state):
    self.state = state

  def begin_epoch(self):
    # Number of training sentences
    self.trained = 0
    # To measure ppl
    self.epoch_ppl = 0
    self.epoch_update_counter = 0
    # To measure time
    self.time = time.time()
    # Verbose
    log.info("Start Epoch %d" % (self.state.finished_epoch))

  def batch_update(self, loss=0, size=1):
    ppl = math.exp(float(loss))
    self.epoch_ppl += ppl
    self.epoch_update_counter += 1
    self.trained += size
    log.info("Sentence trained: %d, Batch-PPL=%f" % (self.trained, ppl))

  def end_epoch(self):
    self.state.finished_epoch += 1
    self.state.time_spent.append(time.time() - self.time)
    self.state.perplexities.append(self.epoch_ppl / self.epoch_update_counter)
    log.info("Epoch %d finished! PPL=%f, time=%d mins" % (self.state.finished_epoch - 1,
                                                          self.state.ppl(),
                                                          self.state.last_time() / 60))
