import chainer
import numpy

import nmtrain
import random

# Environment Variables
xp = None
verbosity = 0
run_mode = nmtrain.enumeration.RunMode.TRAIN
gpu = -1
mem_optimization_level = 0
src_vocab = None
trg_vocab = None

def init(args, run_mode):
  init_run_mode(run_mode)
  init_gpu(args.gpu)
  init_mem_optimization_level(args.memory_optimization)
  if hasattr(args, "seed"):
    if args.seed == 0:
      args.seed = random.randint(1, 1e6)
    init_random(args.seed)
  if hasattr(args, "verbosity"):
    init_verbosity(args.verbosity)

# Environment Functions
def init_gpu(gpu_num):
  global xp, gpu
  xp = numpy
  if hasattr(chainer.cuda, "cupy"):
    if gpu_num >= 0:
      gpu = gpu_num
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

def init_vocabulary(src_voc, trg_voc):
  global src_vocab, trg_vocab
  src_vocab = src_voc
  trg_vocab = trg_voc

def init_run_mode(run_mode_val):
  global run_mode
  run_mode = run_mode_val

def init_mem_optimization_level(opt_level):
  global mem_optimization_level
  mem_optimization_level = opt_level

def array_module():
  assert xp is not None, "Need to call init_gpu first"
  return xp

def is_train():
  return run_mode == nmtrain.enumeration.RunMode.TRAIN

def set_runmode(mode):
  global run_mode
  run_mode = mode

def set_train():
  set_runmode(nmtrain.enumeration.RunMode.TRAIN)

def set_test():
  set_runmode(nmtrain.enumeration.RunMode.TEST)

def memory_optimization_level():
  return mem_optimization_level

def Variable(data):
  volatile = chainer.OFF if is_train() else chainer.ON
  return chainer.Variable(data, volatile=volatile)

def use_gpu():
  global gpu
  return gpu >= 0
