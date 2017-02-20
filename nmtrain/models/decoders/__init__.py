from nmtrain.models.decoders.attentional_lstm_decoder import LSTMAttentionalDecoder
from nmtrain.models.decoders.lstm_decoder import LSTMDecoder

# MISC class for holding the output
class Output(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

