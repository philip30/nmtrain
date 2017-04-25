import chainer
import chainer.functions as F
import numpy

import nmtrain

class EnsembleLinearInterpolateNMT(object):
  def __init__(self, config, models):
    check_ensemble_ok(models)
    self.models = models
      nmtrain.log.warning("len(models) != len(weights) : %d != %d. Uniformly distributing weights" % (len(models), len(config.weight)))
      weights = [1/len(models) for _ in len(models)]
    else:
      if abs(sum(config.weight) - 1) > 1e6:
        nmtrain.log.warning("Sum weights (%f) != 1.0. Uniformly distributing weights" % (sum(config.weight)))
        weights = [1/len(models) for _ in len(models)]
      else:
        weights = config.weight
    self.weights = weights
    self.xp = self.models[0].chainer_model.xp

  def encode(self, src_data):
    for model in self.models:
      model.chainer_model.encode(src_data)

  def update(self, word_var):
    for model in self.models:
      model.chainer_model.update(word_var)

  def set_state(self, state):
    assert(len(self.models) == len(state))
    for model, model_state in zip(self.models, state):
      model.chainer_model.set_state(model_state)

  def decode(self):
    y = 0
    a = 0
    for i, model in enumerate(self.models):
      output = model.chainer_model.decode()
      y += F.scale(output.y, chainer.Variable(self.xp.array(self.weights[i], dtype=numpy.float32), volatile=chainer.ON))
      if hasattr(output, "a"):
        a += F.scale(output.a, chainer.Variable(self.xp.array(self.weights[i], dtype=numpy.float32), volatile=chainer.ON))
      else:
        a = None
    return nmtrain.models.decoders.Output(y=y, a=a)

  def state(self):
    states = []
    for model in self.models:
      states.append(model.chainer_model.state())
    return states

  def set_train(self, value):
    for model in self.models:
      model.chainer_model.set_train(value)

  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    else:
      return getattr(self.models[0], key)

def check_ensemble_ok(models):
  assert(len(models) > 0)
  model = models[0]
  for i in range(1, len(models)):
    other_model = models[i]
    assert model.src_vocab == other_model.src_vocab, "Model-%d src vocabulary does not match" % (i+1)
    assert model.trg_vocab == other_model.trg_vocab, "Model-%d trg vocabulary does not match" % (i+1)
    # Save up some memory by keeping only a single vocabulary object
    del other_model.src_vocab
    del other_model.trg_vocab
    other_model.src_vocab = model.src_vocab
    other_model.trg_vocab = model.trg_vocab
