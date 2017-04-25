import chainer

import nmtrain.environment as environment

# TODO(philip30): Write test for this class
class StackLSTM(chainer.ChainList):
  def __init__(self, in_size, out_size, depth, drop_ratio):
    self.depth = depth
    self.drop_ratio = drop_ratio

    # Init all connections
    lstm = []
    for i in range(depth):
      if i == 0:
        size = in_size
      else:
        size = out_size
      lstm.append(chainer.links.StatelessLSTM(size, out_size))

    # None array for quick reset
    self.none_arr = tuple([None for _ in range(self.depth)])

    # Pass it to the super connection
    super(StackLSTM, self).__init__(*lstm)

  def reset_state(self):
    self.h = self.none_arr
    self.c = self.none_arr

  def set_state(self, h, c):
    self.h = h
    self.c = c

  def __call__(self, x, is_train):
    c, h = [], []
    for i in range(self.depth):
      lstm_in = x if i == 0 else h[i-1]
      c_new, h_new = self[i](self.c[i], self.h[i], lstm_in)
      h_new = chainer.functions.dropout(h_new,
                                        ratio=self.drop_ratio,
                                        train=is_train)
      c_new = chainer.functions.dropout(c_new,
                                        ratio=self.drop_ratio,
                                        train=is_train)
      c.append(c_new)
      h.append(h_new)
    self.c = c
    self.h = h
    return self.h[-1]

  def state(self):
    return (self.h, self.c)

  def set_state(self, state):
    self.h, self.c = state

