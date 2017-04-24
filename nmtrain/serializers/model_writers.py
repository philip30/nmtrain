import nmtrain

class TrainModelWriter(object):
  def __init__(self, out_prefix, save_models):
    self.out_prefix = out_prefix
    self.save_models = save_models

  def save(self, model):
    epoch = model.state.last_trained_epoch
    if self.save_models:
      cols = self.out_prefix.rsplit(".", 1)
      cols[0] += "-%02d" % (epoch)
      if len(cols == 2):
        name = cols[0] + "." + cols[1]
      else:
        name = cols[0] + ".zip"
    else:
      if not self.out_prefix.endswith(".zip"):
        name = self.out_prefix + ".zip"
      else:
        name = self.out_prefix
    nmtrain.serializers.serializer.save(model, name)

