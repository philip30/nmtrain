import chainer

import nmtrain.environment as environment

# TODO(philip30): Write test for this class
class StackLSTM(chainer.ChainList):
  def __init__(self, in_size, out_size, depth, drop_ratio):
    self.depth = depth
    self.drop_ratio = drop_ratio
    self.h = []
    self.c = []
    self.reset_state()

    # Init all connections
    lstm = []
    for i in range(depth):
      if i == 0:
        size = in_size
      else:
        size = out_size
      lstm.append(chainer.links.StatelessLSTM(size, out_size))

    # Pass it to the super connection
    super(StackLSTM, self).__init__(*lstm)

  def reset_state(self):
    del self.h
    del self.c
    self.h = tuple(None for _ in range(self.depth))
    self.c = tuple(None for _ in range(self.depth))

  def set_state(self, h, c):
    self.h = h
    self.c = c

  def __call__(self, x):
    c, h = [], []
    for i in range(len(self.h)):
      lstm_in = x if i == 0 else h[i-1]
      c_new, h_new = self[i](self.c[i], self.h[i], lstm_in)
      h_new = chainer.functions.dropout(h_new,
                                        ratio=self.drop_ratio,
                                        train=environment.is_train())
      c.append(c_new)
      h.append(h_new)
    self.c = tuple(c)
    self.h = tuple(h)
    return self.h[-1]

  def state(self):
    return (self.h, self.c)

  def set_state(self, state):
    self.h, self.c = state

