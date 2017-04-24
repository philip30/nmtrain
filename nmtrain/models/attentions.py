import chainer
import chainer.functions as F

# Not "Defense of the Ancient"
class DotAttentionLayer(chainer.Chain):
  def __call__(self, S, h):
    return F.squeeze(F.softmax(F.batch_matmul(S, h)), axis=2)

class GeneralAttentionLayer(chainer.Chain):
  def __init__(self, hidden_size):
    super(GeneralAttentionLayer, self).__init__(
      inner_weight = chainer.links.Linear(hidden_size, hidden_size)
    )

  def __call__(self, S, h):
    batch_size, src_len, hidden_size = S.data.shape
    S = self.inner_weight(F.reshape(S, (batch_size * src_len, hidden_size)))
    S = F.reshape(S, (batch_size, src_len, hidden_size))
    a = F.softmax(F.squeeze(F.batch_matmul(S, h), axis = 2))
    return a

# MLP layer, as of Bahdanau+ 15
class MLPAttentionLayer(chainer.Chain):
  def __init__(self, hidden_size, attn_size):
    super(MLPAttentionLayer, self).__init__(
      first_layer = chainer.links.Linear(2 * hidden_size, attn_size),
      second_layer = chainer.links.Linear(attn_size, 1)
    )

  def __call__(self, S, h):
    batch_size, src_len, hidden_size = S.data.shape
    h = F.broadcast_to(F.expand_dims(h, axis=2), (batch_size, hidden_size, src_len))
    h = F.swapaxes(h, 1, 2)
    S = F.reshape(F.concat((S, h), axis=2), (batch_size * src_len, 2 * hidden_size))
    a = F.softmax(F.reshape(self.second_layer(F.tanh(self.first_layer(S))), (batch_size, src_len)))
    return a
