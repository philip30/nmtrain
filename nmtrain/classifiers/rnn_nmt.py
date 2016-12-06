import chainer
import numpy
import math

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
           word_penalty=0.0,
           post_processor=None, beam=1):
    if beam >= 1:
      return self.test_beam(model, src_data, watcher,
                            trg_data=trg_data, gen_limit=gen_limit,
                            post_processor=post_processor,
                            store_probabilities=store_probabilities,
                            word_penalty=word_penalty,
                            beam=beam)

    loss, loss_ctr = 0
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
        loss     += nmtrain.chner.cross_entropy(y, y_t)
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
    watcher.end_prediction(loss = loss.data, prediction = prediction,
                           probabilities = probabilities,
                           attention=numpy.transpose(attention))

  def test_beam(self, model, src_data, watcher,
                trg_data=None, gen_limit=50,
                store_probabilities=False,
                post_processor=None, beam=1, word_penalty=0):
    # Exponential distribution of word penalty
    word_penalty = math.exp(word_penalty)
    # The beam used to represent state in beam search
    class BeamState:
      def __init__(self, model_state, prob, word, loss, attention, word_prob, parent):
        self.model_state = model_state
        self.probability = prob
        self.word        = word
        self.attention   = attention
        self.word_prob   = word_prob
        self.parent      = parent
        self.loss        = loss

    # The n-argmax function
    def n_argmax(array, top):
      top = min(top, len(array))
      return numpy.argpartition(array, -top)[-top:]

    # Array module:
    xp = nmtrain.environment.array_module()

    # The beams
    beams = [BeamState(None, 1, None, 0, None, None, None)]
    beam_prediction = []
    worst_prob = 0
    # Start Prediction
    watcher.start_prediction()
    init  = model.encode(src_data)
    for i in range(gen_limit):
      # Expand all the beams
      new_beam = []
      for state in beams:
        if watcher.end_of_sentence(state.word):
          if len(beam_prediction) == 0:
            worst_prob = state.probability
          else:
            worst_prob = min(state.probability, worst_prob)
          beam_prediction.append(state)
        else:
          if state.word is not None:
            model.set_state(state.model_state)
            word_var = nmtrain.environment.Variable(xp.array([state.word], dtype=numpy.int32))
            current_model = model.update(word_var)
          else:
            current_model = init
          # Produce the output
          output = model.decode()
          y_dist = chainer.cuda.to_cpu(output.y.data[0])
          attn_out = chainer.cuda.to_cpu(output.a.data) if hasattr(output, "a") else None
          word_prob = y_dist if store_probabilities else None
          # Produce the next words
          words = n_argmax(y_dist, beam)
          for word in words:
            new_probability = y_dist[word] * word_penalty * state.probability
            new_loss = state.loss - math.log(y_dist[word])
            new_beam.append(BeamState(model_state=current_model, prob=new_probability,
                                      word=word, loss=new_loss, attention=attn_out,
                                      word_prob=word_prob, parent=state))
      # First sort the beam
      new_beam = sorted(new_beam, key = lambda state: state.probability, reverse=True)
      # When the best hypothesis probability is worse than the best probability stop or
      # If no new state is generated
      if len(new_beam) == 0 or new_beam[0].probability < worst_prob:
        break
      else:
        beams = new_beam[:beam]
    # Apparently, no hypothesis reached the end of sentence
    if len(beam_prediction) == 0:
      beam_prediction = [beams[0]]
    else:
      beam_prediction = sorted(beam_prediction, key=lambda state:state.probability, reverse=True)

    # The output of the decoding
    cur_state  = beam_prediction[0]
    attention  = None
    prediction = []
    loss       = cur_state.loss
    while True:
      prediction.append(cur_state.word)
      #TODO(philip30): handle attention
      cur_state = cur_state.parent
      if cur_state.model_state is None:
        break
    loss /= len(prediction)
    watcher.end_prediction(loss = loss, prediction = reversed(prediction),
                           probabilities = None,
                           attention=numpy.transpose(attention))
