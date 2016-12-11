import nmtrain
from . import nmt_encdec

class AttentionalNMT(nmt_encdec.EncoderDecoderNMT):
  def __init__(self, in_size, out_size, hidden_size, embed_size,
               drop_out, lstm_depth, input_feeding, attention_type,
               lexicon):
    super(nmt_encdec.EncoderDecoderNMT, self).__init__(
      encoder = nmtrain.models.encoders.BidirectionalAttentionalEncoder(
        in_size, embed_size, hidden_size, drop_out,
        lstm_depth, input_feeding, lexicon),
      decoder = nmtrain.models.decoders.LSTMAttentionalDecoder(
        out_size, embed_size, hidden_size, drop_out,
        lstm_depth, input_feeding, attention_type, lexicon)
    )

