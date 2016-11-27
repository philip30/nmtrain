from . import encoder_decoder, encoder, decoder

class AttentionalNMT(encoder_decoder.EncoderDecoderNMT):
  def __init__(self, in_size, out_size, hidden_size, embed_size, drop_out, lstm_depth):
    super(encoder_decoder.EncoderDecoderNMT, self).__init__(
      encoder = encoder.BidirectionalAttentionalEncoder(in_size, embed_size,
                                                        hidden_size, drop_out,
                                                        lstm_depth),
      decoder = decoder.LSTMAttentionalDecoder(out_size, embed_size,
                                               hidden_size, drop_out,
                                               lstm_depth)
    )

