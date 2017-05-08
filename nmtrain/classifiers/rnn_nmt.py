import chainer
import numpy
import math
import nmtrain

class RNN_NMT(object):
  def __init__(self):
    self.bptt = None
    self.bptt_len = 0

  def configure_learning(self, learning_config, bptt_func = None):
    self.bptt          = bptt_func
    self.config        = learning_config
    self.learning_type = learning_config.learning.method
    if self.learning_type == "mrt":
      nmtrain.log.info("Setting learning to minimum risk training")
      self.train = self.train_mrt
      self.minrisk   = nmtrain.minrisk.minrisk.MinimumRiskTraining(learning_config.learning.mrt)
    else:
      nmtrain.log.info("Setting learning to maximum likelihood training")
      self.train = self.train_mle

  def set_train(self, is_train):
    self.is_train = is_train
    if hasattr(self, "minrisk"):
      self.minrisk.set_train(is_train)

  def train_mle(self, model, src_batch, trg_batch, eos_id, outputer=None):
    batch_loss  = 0
    bptt_ctr    = 0
    volatile    = chainer.OFF if self.is_train else chainer.ON
    model.encode(src_batch)

    if outputer: outputer.begin_collection(src=src_batch, ref=trg_batch)
    for i, trg_word in enumerate(trg_batch):
      y_t = chainer.Variable(model.xp.array(trg_word, dtype=numpy.int32), volatile=volatile)
      output = model.decode()
      batch_loss += chainer.functions.softmax_cross_entropy(output.y, y_t)
      model.update(y_t)

      # Truncated BPTT
      if self.is_train and self.bptt_len > 0:
        bptt_ctr += 1
        if bptt_ctr == self.config.bptt_len:
          self.bptt(batch_loss)
          bptt_ctr = 0

      if outputer: outputer(output)
    if outputer: outputer.end_collection()
    return batch_loss / len(trg_batch)

  def train_mrt(self, model, src_batch, trg_batch, eos_id, outputer=None):
    return self.minrisk(model, src_batch, trg_batch, eos_id, outputer, self.is_train)

  def predict(self, model, src_sent, eos_id, gen_limit=50,
              store_probabilities=False,
              beam=1, word_penalty=0):
    # The beam used to represent state in beam search
    class BeamState:
      def __init__(self, id, model_state, log_prob, word, attention, word_prob, parent):
        self.id          = id
        self.model_state = model_state
        self.log_prob    = log_prob
        self.word        = word
        self.attention   = attention
        self.word_prob   = word_prob
        self.parent      = parent

    # The n-argmax function
    def n_argmax(array, top):
      top = min(top, len(array))
      return numpy.argpartition(array, -top)[-top:]

    # The beams
    beams = [BeamState(0, None, 0, None, None, None, None)]
    beam_prediction = []
    worst_prob = -float("inf")
    cur_id = 1
    # Start Prediction
    model.encode(src_sent)
    for i in range(gen_limit):
      # Expand all the beams
      new_beam = []
      for state in beams:
        if eos_id == state.word:
          if len(beam_prediction) == 0:
            worst_prob = state.log_prob
          else:
            worst_prob = min(state.log_prob, worst_prob)
          beam_prediction.append(state)
        else:
          if state.word is not None:
            model.set_state(state.model_state)
            word_var = chainer.Variable(model.xp.array([state.word], dtype=numpy.int32), volatile=chainer.ON)
            model.update(word_var)

          # Produce the output
          output = model.decode()
          current_model = model.state()
          y_dist = chainer.cuda.to_cpu(chainer.functions.softmax(output.y).data[0])
          attn_out = chainer.cuda.to_cpu(output.a.data[0]) if hasattr(output, "a") else None
          word_prob = y_dist if store_probabilities else None
          # Produce the next words
          if beam == 1:
            words = [numpy.argmax(y_dist)]
          else:
            words = n_argmax(y_dist, beam)
          for word in words:
            new_probability = math.log(y_dist[word]) + word_penalty + state.log_prob
            new_beam.append(BeamState(id=cur_id, model_state=current_model, log_prob=new_probability,
                                      word=word, attention=attn_out,
                                      word_prob=word_prob, parent=state))
            cur_id += 1
      # First sort the beam
      new_beam = sorted(new_beam, key = lambda state: state.log_prob, reverse=True)
      # When the best hypothesis probability is worse than the best probability stop or
      # If no new state is generated
      if len(new_beam) == 0 or new_beam[0].log_prob < worst_prob:
        break
      else:
        beams = new_beam[:beam]

    # Apparently, no hypothesis reached the end of sentence
    if len(beam_prediction) == 0:
      beam_prediction = [beams[0]]
    else:
      beam_prediction = sorted(beam_prediction,
                               key=lambda state:state.log_prob,
                               reverse=True)

    ## Collecting output
    cur_state  = beam_prediction[0]
    # attention
    attention_available = cur_state.attention is not None
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
    # Output: Word probabilities
    if store_probabilities:
      output.probabilities = numpy.concatenate(list(reversed(probabilities)), axis=1)
    return output

