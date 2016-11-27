import chainer
import nmtrain

# Relative import
from . import encoder, decoder

class EncoderDecoderNMT(chainer.Chain):
  def __init__(self, in_size, out_size, hidden_size, embed_size, drop_out, lstm_depth):
    super(EncoderDecoderNMT, self).__init__(
        encoder = encoder.BidirectionalEncoder(in_size, embed_size,
                                               hidden_size, drop_out,
                                               lstm_depth, attention=False),
        decoder = decoder.LSTMDecoder(out_size, embed_size,
                                      hidden_size, drop_out,
                                      lstm_depth)
    )

  def encode(self, src_data):
    """ Encode the whole source sentence into a single representation 'h' """
    self.decoder.init(self.encoder(src_data))

  def decode(self):
    """ Produces the word probability distribution based on the last state of decoder """
    return chainer.functions.softmax(self.decoder())

  def update(self, word):
    """ Produces the next state depends on current word """
    self.decoder.update(word)

