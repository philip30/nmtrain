import nmtrain
import numpy

class NmtrainState(object):
  def __init__(self):
    self.data = nmtrain.state_pb.NmtrainState()

  @property
  def last_trained_epoch(self):
    return self.data.train_state.finished_epoch

  def is_ppl_decline(self):
    dev_updates = self.data.dev_updates
    if len(dev_updates) < 2:
      return False
    dev_ppl_now  = dev_updates[-1].score["ppl"]
    dev_ppl_prev = dev_updates[-2].score["ppl"]

    if dev_ppl_now == 0 and dev_ppl_prev == 0:
      return False

    return dev_ppl_now > dev_ppl_prev

  def is_ppl_minimum(self):
    dev_ppls = [dev_update.score["ppl"] for dev_update in self.data.dev_updates]
    return numpy.argmin(dev_ppl) == len(dev_ppls) - 1

  def new_epoch(self):
    self.data.train_state.finished_epoch += 1
