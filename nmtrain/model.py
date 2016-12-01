import chainer
import sys

import nmtrain
import nmtrain.models
import nmtrain.log as log

class NmtrainModel:
  """
  Class to represent the model being saved to the file.
  Contains:
      - Vocabularies
      - Chainer model (parameters + real computation) -> nmtrain/models/...
      - Training State
  """
  def __init__(self, args):
    # Init Model
    if args.init_model:
      nmtrain.serializer.load(self, args.init_model)
    else:
      self.src_vocab = nmtrain.Vocabulary(True, True, True)
      self.trg_vocab = nmtrain.Vocabulary(True, True, True)
      self.optimizer = parse_optimizer(args.optimizer)
      self.training_state = TrainingState()
      self.specification = args
      self.chainer_model = None

  def finalize_model(self, args):
    if self.chainer_model is None:
      self.chainer_model = from_spec(args, len(self.src_vocab), len(self.trg_vocab))
      self.optimizer.setup(self.chainer_model)

    if hasattr(self, "optimizer"):
      self.optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradient_clipping))
      # TODO(philip30): Disable it for a time being
      #self.optimizer.add_hook(chainer.optimizer.GradientNoise(args.gradient_noise))

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      self.chainer_model.to_gpu(nmtrain.environment.gpu)

  def describe(self):
    print("~ Nmtrain ~", file=sys.stderr)
    print("~ By: Philip Arthur (philip.arthur30@gmail.com)", file=sys.stderr)
    print("Model Type     :", self.specification.model_architecture, file=sys.stderr)
    print("Hidden Size    :", self.specification.hidden, file=sys.stderr)
    print("Embed Size     :", self.specification.embed, file=sys.stderr)
    print("LSTM Depth     :", self.specification.depth, file=sys.stderr)
    print("SRC Vocab Size :", len(self.src_vocab), file=sys.stderr)
    print("TRG Vocab Size :", len(self.trg_vocab), file=sys.stderr)
    print("Dropout Ratio  :", self.specification.dropout, file=sys.stderr)
    print("Unknown Cut    :", self.specification.unk_cut, file=sys.stderr)
    print("Batch Size     :", self.specification.batch, file=sys.stderr)
    if hasattr(self, "optimizer"):
      print("Optimizer      :", self.optimizer.__class__.__name__, file=sys.stderr)
    print("Finished Iters :", self.training_state.finished_epoch, file=sys.stderr)

class TrainingState(object):
  def __init__(self):
    self.finished_epoch   = 0
    self.perplexities     = []
    self.dev_perplexities = []
    self.bleu_scores      = []
    self.time_spent       = []
    self.wps_time         = []
    self.batch_indexes    = None

  def ppl(self):
    return self.perplexities[-1]

  def dev_ppl(self):
    return self.dev_perplexities[-1]

  def bleu(self):
    return self.bleu_scores[-1]

  def last_time(self):
    return self.time_spent[-1]

  def wps(self):
    return self.wps_time[-1]

  def time(self):
    return sum(self.time_spent)

class TestState(TrainingState):
  def __init__(self):
    self.wps_time    = []
    self.time_spent  = []
    self.bleu_scores = []
    self.perplexities = []

def parse_optimizer(optimizer_str):
  optimizer_str = optimizer_str.split(":")
  opt = optimizer_str[0]
  if len(optimizer_str) == 1:
    opt_param = ""
  else:
    opt_param = optimizer_str[1]
  # Select optimizer
  if opt == "adam":
    param = parse_parameter(opt_param, {
      "alpha":float, "beta1":float, "beta2": float, "eps": float})
    return chainer.optimizers.Adam(**param)
  elif opt == "sgd":
    param = parse_parameter(opt_param, {
      "lr": float})
    return chainer.optimizers.SGD(**param)
  else:
    raise ValueError("Unrecognized optimizer:", opt)

def parse_parameter(opt_param, param_mapping):
  if len(opt_param) == 0:
    return {}
  param = {}
  for param_str in opt_param.split(","):
    param_str = param_str.split("=")
    assert len(param_str) == 2, "Bad parameter line: %s" % (opt_param)
    if param_str[0] not in param_mapping:
      raise ValueError("Unrecognized parameter:", param_str)
    else:
      param[param_str[0]] = param_mapping[param_str[0]](param_str[1])
  return param

def from_spec(spec, in_size, out_size):
  if spec.model_architecture == "encdec":
    return nmtrain.models.EncoderDecoderNMT(
      embed_size   = spec.embed,
      hidden_size  = spec.hidden,
      drop_out     = spec.dropout,
      lstm_depth   = spec.depth,
      in_size      = in_size,
      out_size     = out_size
    )
  elif spec.model_architecture == "attn":
    return nmtrain.models.AttentionalNMT(
      embed_size   = spec.embed,
      hidden_size  = spec.hidden,
      drop_out     = spec.dropout,
      lstm_depth   = spec.depth,
      in_size      = in_size,
      out_size     = out_size,
      input_feeding = spec.input_feeding
    )
  else:
    raise Exception("Unknown Model Type:", model_type)

