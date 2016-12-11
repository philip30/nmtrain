import nmtrain

import chainer

def chainer_mem_optimize(func, args, level, kwargs={}):
  if type(args) != list and type(args) != tuple:
    args = (args,)
  if level >= nmtrain.environment.memory_optimization_level():
    return chainer.functions.forget(func, *args, **kwargs)
  else:
    return func(*args)
