import nmtrain

from nmtrain.models import nmt_encdec
from nmtrain.models import encoders
from nmtrain.models import decoders

class AttentionalNMT(nmt_encdec.EncoderDecoderNMT):
  def __init__(self, in_size, out_size, hidden_units, drop_out, lstm_depth, input_feeding, attention_type, lexicon):
    super(nmt_encdec.EncoderDecoderNMT, self).__init__()
    with self.init_scope():
      self.encoder = nmtrain.models.encoders.BidirectionalAttentionalEncoder(
                                 in_size, hidden_units,
                                 drop_out, lstm_depth, input_feeding, lexicon)
      self.decoder = nmtrain.models.decoders.LSTMAttentionalDecoder(
                                 out_size, hidden_units,
                                 drop_out, lstm_depth, input_feeding,
                                 attention_type, lexicon)

