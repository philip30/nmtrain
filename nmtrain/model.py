import chainer
import sys

import nmtrain
import nmtrain.models
import nmtrain.log as log

# This spec attribute define the number of parameters of network.
# It should not be changed. If model is loaded, then these settings will be 
# loaded from the previous model
OVERWRITE_SPEC = ["hidden", "embed", "depth", "model_architecture", "batch",
                  "unk_cut", "dropout", "src_max_vocab", "trg_max_vocab", "max_sent_length",
                  "init_model", "seed", "attention_type", "input_feeding", "lexicon_method",
                  "lexicon_alpha"]

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
      for spec in OVERWRITE_SPEC:
        if hasattr(self.specification, spec):
          setattr(args, spec, getattr(self.specification, spec))
      self.name = args.init_model
    else:
      self.src_vocab = nmtrain.Vocabulary(True, True)
      self.trg_vocab = nmtrain.Vocabulary(True, True)
      self.optimizer = parse_optimizer(args.optimizer)
      self.training_state = TrainingState()
      self.specification = args
      self.chainer_model = None
      self.lexicon = None
      self.bpe_codec = load_bpe_codec(args.src_bpe_codec, args.trg_bpe_codec)

    if hasattr(self, "optimizer"):
      self.optimizer.use_cleargrads()

    if hasattr(self, "bpe_codec"):
      nmtrain.environment.init_bpe_codec(self.bpe_codec)

  def finalize_model(self):
    if self.lexicon is None and hasattr(self.specification, "lexicon") and self.specification.lexicon:
      self.lexicon = nmtrain.Lexicon(self.specification.lexicon,
                                     self.src_vocab, self.trg_vocab,
                                     self.specification.lexicon_alpha,
                                     self.specification.lexicon_method)
    else:
      self.lexicon = None

    if self.chainer_model is None:
      self.chainer_model = from_spec(self.specification, self.src_vocab, self.trg_vocab, self.lexicon)
      self.optimizer.setup(self.chainer_model)

    if hasattr(self, "optimizer"):
      self.optimizer.add_hook(chainer.optimizer.GradientClipping(self.specification.gradient_clipping))

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      self.chainer_model.to_gpu(nmtrain.environment.gpu)

  def describe(self):
    print("~ NMTrain-Model ~", file=sys.stderr)
    print("Model Type     :", self.specification.model_architecture, file=sys.stderr)
    print("Hidden Size    :", self.specification.hidden, file=sys.stderr)
    print("Embed Size     :", self.specification.embed, file=sys.stderr)
    print("LSTM Depth     :", self.specification.depth, file=sys.stderr)
    print("SRC Vocab Size :", len(self.src_vocab) - 2, file=sys.stderr)
    print("TRG Vocab Size :", len(self.trg_vocab) - 2, file=sys.stderr)
    print("Dropout Ratio  :", self.specification.dropout, file=sys.stderr)
    print("Unknown Cut    :", self.specification.unk_cut, file=sys.stderr)
    print("Batch Size     :", self.specification.batch, file=sys.stderr)
    if hasattr(self, "optimizer"):
      print("Optimizer      :", self.optimizer.__class__.__name__, file=sys.stderr)
    print("Finished Iters :", self.training_state.finished_epoch, file=sys.stderr)
    print("Trained Sentences:", self.training_state.trained_sentence, file=sys.stderr)

  @property
  def xp(self):
    return self.chainer_model.xp

class TrainingState(object):
  def __init__(self):
    self.finished_epoch   = 0
    self.trained_sentence = 0
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

def load_bpe_codec(src_bpe_codec, trg_bpe_codec):
  if len(src_bpe_codec) > 0 and len(trg_bpe_codec) > 0:
    src_codec = nmtrain.bpe.BPE(src_bpe_codec)
    trg_codec = nmtrain.bpe.BPE(trg_bpe_codec)
    return src_codec, trg_codec
  else:
    return None, None

def from_spec(spec, src_voc, trg_voc, lexicon):
  in_size, out_size = len(src_voc), len(trg_voc)
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
      embed_size     = spec.embed,
      hidden_size    = spec.hidden,
      drop_out       = spec.dropout,
      lstm_depth     = spec.depth,
      in_size        = in_size,
      out_size       = out_size,
      input_feeding  = spec.input_feeding,
      attention_type = spec.attention_type,
      # (Arthur et al., 2016)
      lexicon        = lexicon,
    )
  else:
    raise Exception("Unknown Model Type:", model_type)

