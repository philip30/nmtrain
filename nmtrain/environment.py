import chainer
import numpy

import nmtrain
import random

if chainer.cuda.available:
  import cupy

# Environment Variables
verbosity = 0
gpu = -1

def init(args):
  if hasattr(args, "gpu"):
    init_gpu(args.gpu)
  if hasattr(args, "seed"):
    if args.seed == 0:
      args.seed = random.randint(1, 1e6)
    init_random(args.seed)
  if hasattr(args, "verbosity"):
    init_verbosity(args.verbosity)

# Environment Functions
def init_gpu(gpu_num):
  global gpu
  if chainer.cuda.available and gpu_num >= 0:
    gpu = gpu_num
    chainer.cuda.get_device(gpu_num).use()
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
