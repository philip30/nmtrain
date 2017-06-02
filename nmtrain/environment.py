import chainer
import numpy

import nmtrain
import random

if chainer.cuda.available:
  import cupy

# Environment Variables
verbosity = 0
gpu = -1

def init(config, args):
  if hasattr(config, "gpu"):
    init_gpu(config.gpu)
  if hasattr(config, "seed"):
    if config.seed == 0:
      config.seed = random.randint(1, 1e6)
    init_random(config.seed)
  if hasattr(config, "verbosity"):
    init_verbosity(config.verbosity)
  if not hasattr(args, "debug") or not args.debug:
    chainer.config.debug = False
    chainer.config.type_check = False

# Environment Functions
def init_gpu(gpu_num):
  global gpu
  if chainer.cuda.available and gpu_num >= 0:
    gpu = gpu_num
    chainer.cuda.get_device(gpu_num).use()
  if chainer.cuda.cudnn_enabled:
    chainer.config.use_cudnn = True
    chainer.config.cudnn_deterministic = True
  return gpu_num

def init_random(seed):
  if seed != 0:
    if chainer.cuda.available:
      cupy.random.seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def init_verbosity(verbosity_level):
  global verbosity
  verbosity = verbosity_level

def use_gpu():
  return gpu >= 0
