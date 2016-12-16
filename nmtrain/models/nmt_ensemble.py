import chainer.functions as F
import numpy

import nmtrain

class EnsembleLinearInterpolateNMT(object):
  def __init__(self, models):
    check_ensemble_ok(models)
    self.models = models

    xp = nmtrain.environment.array_module()
    self.normalization_constant = xp.array(1.0 / len(self.models), dtype=numpy.float32)

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
    arr_sum = None
    a = None
    for i, model in enumerate(self.models):
      output = model.chainer_model.decode()
      if i == 0:
        arr_sum = output.y
        if hasattr(output, "a"): a = output.a
      else:
        arr_sum += output.y
    prob = F.scale(arr_sum, nmtrain.environment.Variable(self.normalization_constant))
    return nmtrain.models.decoders.Output(y=prob, a=a)

  def state(self):
    states = []
    for model in self.models:
      states.append(model.chainer_model.state())
    return states

  def __getattr__(self, key):
    if key in self.__dict__:
      return self.__dict__[key]
    else:
      return getattr(self.models[0], key)

class EnsembleLogSumNMT(EnsembleLinearInterpolateNMT):
  def decode(self):
    arr_sum = None
    a = None
    for i, model in enumerate(self.models):
      output = model.chainer_model.decode()
      output.y = F.log(output.y)
      if i == 0:
        arr_sum = output.y
        if hasattr(output, "a"): a = output.a
      else:
        arr_sum += output.y
    prob = F.exp(F.scale(arr_sum, nmtrain.environment.Variable(self.normalization_constant)))
    return nmtrain.models.decoders.Output(y=prob, a=a)

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
