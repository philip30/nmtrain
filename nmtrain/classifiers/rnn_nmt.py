import chainer
import numpy

import nmtrain.chner
import nmtrain.environment

class RNN_NMT(object):
  """ Recurrent neural network neural machine translation"""
  def train(self, model, src_data, trg_data, watcher, bptt, bptt_len=0):
    batch_loss  = 0
    bptt_ctr    = 0
    model.encode(src_data)
    for trg_word in trg_data:
      y_t = nmtrain.environment.Variable(trg_word)
      y   = model.decode()

      batch_loss += nmtrain.chner.cross_entropy(y, y_t)
      model.update(y_t)

      # Truncated BPTT
      if bptt_len > 0:
        bptt_ctr += 1
        if bptt_ctr == bptt_len:
          bptt(batch_loss)
          bptt_ctr = 0

    batch_loss /= len(trg_data)
    return batch_loss

  def test(self, model, src_data, watcher, trg_data=None, gen_limit=1, store_probabilities=False, force_limit=False):
    xp     = nmtrain.environment.array_module()
    argmax = chainer.functions.argmax
    loss, loss_ctr = 0, 0
    prediction     = []
    probabilities  = []
    if trg_data is not None:
      gen_limit = len(trg_data)

    # Start Prediction
    watcher.start_prediction()
    model.encode(src_data)
    for i in range(gen_limit):
      # Generate word output probability distribution
      y = model.decode()
      word_var = chainer.functions.argmax(y)
      model.update(chainer.functions.reshape(word_var, (1,)))
      # Whether to store softmax probability
      if store_probabilities:
        probabilities.append(chainer.cuda.to_cpu(y.data[0]))
      # Calculate Perplexity
      if trg_data is not None:
        y_t       = nmtrain.environment.Variable(trg_data[i])
        loss     += float(nmtrain.chner.cross_entropy(y, y_t).data)
        loss_ctr += 1
      word = numpy.asscalar(chainer.cuda.to_cpu(word_var.data))
      prediction.append(word)
      if watcher.end_of_sentence(word) and not force_limit:
        break
    loss /= loss_ctr
    watcher.end_prediction(loss = loss, prediction = prediction, probabilities = probabilities)
