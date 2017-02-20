import chainer
import nmtrain

class EncoderDecoderNMT(chainer.Chain):
  def __init__(self, in_size, out_size, hidden_size, embed_size, drop_out, lstm_depth):
    super(EncoderDecoderNMT, self).__init__(
        encoder = nmtrain.models.encoders.BidirectionalEncoder(in_size, embed_size,
                                                               hidden_size, drop_out,
                                                               lstm_depth),
        decoder = nmtrain.models.decoders.LSTMDecoder(out_size, embed_size,
                                                      hidden_size, drop_out,
                                                      lstm_depth)
    )

  def encode(self, src_data):
    """ Encode the whole source sentence into a single representation 'h' """
    return self.decoder.init(self.encoder(src_data))

  def decode(self):
    """ Produces the word probability distribution based on the last state of decoder """
    return self.decoder()

  def update(self, word_var):
    """ Produces the next state depends on current word """
    return self.decoder.update(word_var)

  def set_state(self, state):
    self.decoder.set_state(state)

  def state(self):
    return self.decoder.state()

