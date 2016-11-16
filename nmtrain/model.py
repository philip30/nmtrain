import chainer

import nmtrain
import nmtrain.models

class NmtrainModel:
  """
  Class to represent the model being saved to the file.
  Contains:
      - Vocabularies
      - Chainer model (parameters + real computation) -> nmtrain/models/...
      - Training State
  """
  def __init__(self, args):
    self.src_vocab = nmtrain.Vocabulary(True, True, True)
    self.trg_vocab = nmtrain.Vocabulary(True, True, True)
    self.training_state = TrainingState()
    self.optimizer = parse_optimizer(args.optimizer)
   # Init Model
    if args.init_model:
      pass # TODO(philip30): Implement init model
    else:
      self.chainer_model = None

  def finalize_model(self, args):
    # Construct appropriate model if model has not been initialized
    if self.chainer_model is None:
      if args.model_architecture == "encdec":
        self.chainer_model = nmtrain.models.EncoderDecoderNMT(
          embed_size   = args.embed,
          hidden_size  = args.hidden,
          drop_out     = args.dropout,
          lstm_depth   = args.depth,
          in_size      = len(self.src_vocab),
          out_size     = len(self.trg_vocab)
        )
      elif args.model_architecture == "attn":
        self.chainer_model = nmtrain.models.AttentionalNMT(
          embed_size   = args.embed,
          hidden_size  = args.hidden,
          drop_out     = args.dropout,
          lstm_depth   = args.depth,
          in_size      = len(self.src_vocab),
          out_size     = len(self.trg_vocab)
        )
      else:
        raise Exception("Unknown Model Type:", model_type)

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      self.chainer_model.to_gpu(nmtrain.environment.gpu)

    # Setup Optimizer
    if self.optimizer is not None:
      self.optimizer.setup(self.chainer_model)
      # TODO(philip30): Implement optimizer loading here

class TrainingState(object):
  def __init__(self):
    self.finished_epoch   = 0
    self.perplexities     = []
    self.dev_perplexities = []
    self.bleu_scores      = []
    self.time_spent       = []
    self.wps              = []
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
    retirm self.wps[-1]

  def time(self):
    return sum(self.time_spent)

def parse_optimizer(optimizer_str):
  opt, opt_param = optimizer_str.split(":")
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
  param = {}
  for param_str in opt_param.split(","):
    param_str = param_str.split("=")
    assert len(param_str) == 2, "Bad parameter line: %s" % (opt_param)
    if param_str[0] not in param_mapping:
      raise ValueError("Unrecognized parameter:", param_str)
    else:
      param[param_str[0]] = param_mapping[param_str[0]](param_str[1])
  return param

