import chainer

import nmtrain

class LSTMDecoder(chainer.Chain):
  def __init__(self, out_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(LSTMDecoder, self).__init__(
      decoder       = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
      affine_vocab  = chainer.links.Linear(hidden_size, out_size),
      output_embed  = chainer.links.EmbedID(out_size, embed_size),
      state_init    = chainer.links.Linear(hidden_size, embed_size)
    )

  def __call__(self):
    mem_optimize = nmtrain.optimization.chainer_mem_optimize
    y = mem_optimize(chainer.functions.softmax,
                     mem_optimize(self.affine_vocab, chainer.functions.tanh(self.h), level=1), level=1)
    return nmtrain.models.decoders.Output(y=y)

  def init(self, h):
    self.decoder.reset_state()
    self.h = self.decoder(self.state_init(h))

  def update(self, next_word):
    self.h = self.decoder(self.output_embed(next_word))

  def set_state(self, state):
    self.h, state = state
    self.decoder.set_state(state)

  def state(self):
    return self.h, self.decoder.state()

