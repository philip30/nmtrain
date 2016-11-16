import chainer

import nmtrain

class LSTMDecoder(chainer.Chain):
  def __init__(self, out_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(LSTMDecoder, self).__init__(
      decoder       = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
      affine_vocab  = chainer.links.Linear(hidden_size, out_size),
      output_embed  = chainer.links.EmbedID(out_size, embed_size)
    )

  def init(self, h):
    self.decoder.reset_state()
    self.decoder(h)

  def __call__(self, h):
    return self.affine_vocab(chainer.functions.tanh(h))

  def update(self, next_word):
    self.decoder(self.output_embed(next_word))

  def reset_state(self):
    self.decoder.reset_state()
