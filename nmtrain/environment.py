import chainer
import numpy

import nmtrain
import nmtrain.enum

# Environment Variables
xp = None
verbosity = 0
run_mode = nmtrain.enum.RunMode.TRAIN

# Environment Functions
def init_gpu(gpu_num):
  global xp
  if not hasattr(chainer.cuda, "cupy"):
    gpu_num = -1
  if gpu_num >= 0:
    xp = chainer.cuda.cupy
    chainer.cuda.get_device(gpu_num).use()
  else:
    xp = numpy
  return gpu_num

def init_random(seed):
  if seed != 0:
    if hasattr(chainer.cuda, "cupy"):
      chainer.cuda.cupy.random.seed(seed)
    numpy.random.seed(seed)

def init_verbosity(verbosity_level):
  global verbosity
  verbosity = verbosity_level

def init_run_mode(run_mode_val):
  global run_mode
  run_mode = run_mode_val

def array_module():
  assert xp is not None, "Need to call init_gpu first"
  return xp

def is_train():
  return run_mode == nmtrain.enum.RunMode.TRAIN