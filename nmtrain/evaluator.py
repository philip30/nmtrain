class TrainingEvaluator(object):
  def __init__(self, state):
    self.state = state

  def should_early_stop(self):
    return False

  def should_save(self):
    return True
