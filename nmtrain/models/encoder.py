import chainer
import chainer.functions as F
import numpy

import nmtrain
import nmtrain.chner
import nmtrain.environment

class BidirectionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(BidirectionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(2*hidden_size, embed_size)
    )

  def __call__(self, src_data):
    # Reset both encoders
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()

    # Perform encoding
    fe = None
    for j in range(len(src_data)):
      fe = self.encode_forward(self.embed(nmtrain.environment.Variable(src_data[j])))
      be = self.encode_backward(self.embed(nmtrain.environment.Variable(src_data[-j-1])))

    return self.encode_project(F.concat((fe, be), axis=1))

class BidirectionalAttentionalEncoder(chainer.Chain):
  def __init__(self, in_size, embed_size, hidden_size, dropout_ratio, lstm_depth):
    super(BidirectionalAttentionalEncoder, self).__init__(
        embed           = chainer.links.EmbedID(in_size, embed_size),
        encode_forward  = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_backward = nmtrain.chner.StackLSTM(embed_size, hidden_size, lstm_depth, dropout_ratio),
        encode_project  = chainer.links.Linear(2*hidden_size, embed_size)
    )

  def __call__(self, src_data):
    self.encode_forward.reset_state()
    self.encode_backward.reset_state()
 
    # Perform encoding
    fe, be = [], []
    for j in range(len(src_data)):
      fe.append(self.encode_forward(self.embed(nmtrain.environment.Variable(src_data[j]))))
      be.append(self.encode_backward(self.embed(nmtrain.environment.Variable(src_data[-j-1]))))

    # Joining encoding together
    S = []
    for i in range(len(fe)):
      h = self.encode_project(F.concat((fe[i], be[-1-i]), axis=1))
      S.append(F.expand_dims(h, axis=2))
    S = F.swapaxes(F.concat(S, axis=2), 1, 2)

    # Append 0 to the end of h
    xp = nmtrain.environment.array_module()
    shape = fe[0].data.shape # (batch_size, hidden_size)
    zero = nmtrain.environment.Variable(xp.zeros(shape, dtype=numpy.float32))
    h = F.hstack((h, zero))
 
    return h, S

