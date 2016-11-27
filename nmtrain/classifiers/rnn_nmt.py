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
      output = model.decode()

      batch_loss += nmtrain.chner.cross_entropy(output.y, y_t)
      model.update(y_t)

      # Truncated BPTT
      if bptt_len > 0:
        bptt_ctr += 1
        if bptt_ctr == bptt_len:
          bptt(batch_loss)
          bptt_ctr = 0

    batch_loss /= len(trg_data)
    return batch_loss

  def test(self, model, src_data, watcher,
           trg_data=None, gen_limit=50,
           store_probabilities=False, force_limit=False,
           post_processor=None):
    loss, loss_ctr = 0, 0
    prediction     = []
    probabilities  = []
    attention      = None
    if trg_data is not None:
      gen_limit = len(trg_data)

    # Start Prediction
    watcher.start_prediction()
    model.encode(src_data)
    for i in range(gen_limit):
      # Generate word output probability distribution
      output = model.decode()
      y      = output.y
      word_var = chainer.functions.argmax(y)
      model.update(chainer.functions.reshape(word_var, (1,)))
      # Whether to store softmax probability
      if store_probabilities:
        probabilities.append(chainer.cuda.to_cpu(y.data[0]))
      # Check if attention is also outputted
      if hasattr(output, "a"):
        attention_vec = chainer.cuda.to_cpu(output.a.data)
        if attention is None:
          attention = attention_vec
        else:
          attention = numpy.concatenate((attention, attention_vec), axis=0)
      # Calculate Perplexity
      if trg_data is not None:
        y_t       = nmtrain.environment.Variable(trg_data[i])
        loss     += float(nmtrain.chner.cross_entropy(y, y_t).data)
        loss_ctr += 1
      # Convert to word
      word = numpy.asscalar(chainer.cuda.to_cpu(word_var.data))
      prediction.append(word)
      # Should we stop now
      if watcher.end_of_sentence(word) and not force_limit:
        break
    if loss_ctr != 0:
      loss /= loss_ctr
    # TODO(philip30): Implement PostProcessor
    watcher.end_prediction(loss = loss, prediction = prediction,
                           probabilities = probabilities,
                           attention=numpy.transpose(attention))

