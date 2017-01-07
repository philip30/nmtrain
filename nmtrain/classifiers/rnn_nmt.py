import chainer
import numpy
import math

import nmtrain.chner

class RNN_NMT(object):
  """ Recurrent neural network neural machine translation"""
  def train(self, model, src_batch, trg_batch, bptt, bptt_len=0):
    batch_loss  = 0
    bptt_ctr    = 0
    model.encode(src_batch.data)

    for trg_word in trg_batch.data:
      y_t = nmtrain.environment.VariableArray(model, trg_word)
      output = model.decode()
      batch_loss += nmtrain.chner.cross_entropy(output.y, y_t)
      model.update(y_t)

      # Truncated BPTT
      if bptt_len > 0:
        bptt_ctr += 1
        if bptt_ctr == bptt_len:
          bptt(batch_loss)
          bptt_ctr = 0

    return batch_loss / len(trg_batch.data)

  def eval(self, model, src_sent, trg_sent):
    loss = 0
    # Start Prediction
    model.encode(src_sent.data)
    for trg_word in trg_sent.data:
      y_t    = nmtrain.environment.VariableArray(model, trg_word)
      output = model.decode()
      loss  += nmtrain.chner.cross_entropy(output.y, y_t)
      model.update(y_t)
    return float(loss.data) / len(trg_sent.data)

  def predict(self, model, src_sent, eos_id, gen_limit=50,
              store_probabilities=False,
              beam=1, word_penalty=0):
    # Exponential distribution of word penalty
    word_penalty = math.exp(word_penalty)
    # The beam used to represent state in beam search
    class BeamState:
      def __init__(self, id, model_state, prob, word, attention, word_prob, parent):
        self.id          = id
        self.model_state = model_state
        self.probability = prob
        self.word        = word
        self.attention   = attention
        self.word_prob   = word_prob
        self.parent      = parent

      def __str__(self):
        return ", ".join([str(self.id), nmtrain.environment.trg_vocab.word(self.word), str(self.probability), str(self.parent.id)])

    # The n-argmax function
    def n_argmax(array, top):
      top = min(top, len(array))
      return numpy.argpartition(array, -top)[-top:]

    # The beams
    beams = [BeamState(0, None, 1, None, None, None, None)]
    beam_prediction = []
    worst_prob = 0
    cur_id = 1
    # Start Prediction
    model.encode(src_sent.data)
    for i in range(gen_limit):
      # Expand all the beams
      new_beam = []
      for state in beams:
        if eos_id == state.word:
          if len(beam_prediction) == 0:
            worst_prob = state.probability
          else:
            worst_prob = min(state.probability, worst_prob)
          beam_prediction.append(state)
        else:
          if state.word is not None:
            model.set_state(state.model_state)
            word_var = nmtrain.environment.Variable(model.xp.array([state.word], dtype=numpy.int32))
            model.update(word_var)

          # Produce the output
          output = model.decode()
          current_model = model.state()
          y_dist = chainer.cuda.to_cpu(output.y.data[0])
          attn_out = chainer.cuda.to_cpu(output.a.data[0]) if hasattr(output, "a") else None
          word_prob = y_dist if store_probabilities else None
          # Produce the next words
          if beam == 1:
            words = [numpy.argmax(y_dist)]
          else:
            words = n_argmax(y_dist, beam)
          for word in words:
            new_probability = y_dist[word] * word_penalty * state.probability
            new_beam.append(BeamState(id=cur_id, model_state=current_model, prob=new_probability,
                                      word=word, attention=attn_out,
                                      word_prob=word_prob, parent=state))
            cur_id += 1
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
      beam_prediction = sorted(beam_prediction,
                               key=lambda state:state.probability,
                               reverse=True)

    ## Collecting output
    cur_state  = beam_prediction[0]
    # attention
    attention_available = hasattr(cur_state, "attention")
    attention = [] if attention_available else None
    # probability of each time step
    probabilities = [] if store_probabilities else None
    # Prediction
    prediction = []
    while cur_state.parent is not None:
      prediction.append(cur_state.word)
      if attention_available:
        attention.append(numpy.expand_dims(cur_state.attention, axis=1))
      if store_probabilities:
        probabilities.append(numpy.expand_dims(cur_state.word_prob, axis=1))
      cur_state = cur_state.parent
    ## Packing output
    output = lambda: None
    output.prediction = list(reversed(prediction))
    # Output: Attention
    if attention_available:
      output.attention = numpy.concatenate(list(reversed(attention)), axis=1)
    else:
      output.attention = None
    # Output: Word probabilities
    if store_probabilities:
      output.probabilities = numpy.concatenate(list(reversed(probabilities)), axis=1)
    else:
      output.probabilities = None
    return output
