import chainer
import nmtrain

class NmtrainModel(object):
  def __init__(self, config):
    self.config = config
    # Init Model
    if hasattr(config, "init_model") and config.init_model:
      nmtrain.serializer.load(self, config.init_model)
    else:
      self.src_vocab = nmtrain.Vocabulary(True, True)
      self.trg_vocab = nmtrain.Vocabulary(True, True)
      self.optimizer = parse_optimizer(config.learning_config.optimizer)
      self.state     = nmtrain.NmtrainState()
      self.chainer_model = None
      self.lexicon       = None
      self.bpe_codec     = load_bpe_codec(config.bpe_config)

    if hasattr(self, "optimizer"):
      self.optimizer.use_cleargrads()

  def finalize_model(self):
    if self.lexicon is None and hasattr(self.config, "lexicon_config"):
      if self.config.lexicon_config.path:
        self.lexicon = nmtrain.Lexicon(self.config.lexicon_config.path,
                                       self.src_vocab, self.trg_vocab,
                                       self.config.lexicon_config.lexicon_alpha,
                                       self.config.lexicon_config.method)
    else:
      self.lexicon = None

    if self.chainer_model is None:
      self.chainer_model = from_spec(self.config.network_config, self.config.learning_config,
                                     self.src_vocab, self.trg_vocab, self.lexicon)
      self.optimizer.setup(self.chainer_model)

      # Initializing model
      initializer = chainer.initializers.Uniform(scale=0.1)
      for name, array in sorted(self.chainer_model.namedparams()):
        initializer(array.data)

    if hasattr(self, "optimizer"):
      self.optimizer.add_hook(chainer.optimizer.GradientClipping(self.config.learning_config.gradient_clipping))

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      self.chainer_model.to_gpu(nmtrain.environment.gpu)

  def describe(self):
    pass

  @property
  def xp(self):
    return self.chainer_model.xp

## ROUTINES
def parse_optimizer(optimizer):
  opt = optimizer.type
  param = optimizer.params
  # Select optimizer
  if opt == "adam":
    return chainer.optimizers.Adam(alpha=param.adam.alpha,
                                   beta1=param.adam.beta1,
                                   beta2=param.adam.beta2,
                                   eps=param.adam.eps)
  elif opt == "sgd":
    return chainer.optimizers.SGD(lr=param.adam.lr)
  else:
    raise ValueError("Unrecognized optimizer:", opt)

def load_bpe_codec(config):
  if hasattr(config, "bpe_config") and \
      config.bpe_config.src_codec and \
      config.bpe_config.trg_codec:
    return  nmtrain.bpe.BPE(config.bpe_config.src_codec), \
            nmtrain.bpe.BPE(config.bpe_config.trg_codec)
  else:
    return None, None

def from_spec(network_config, learning_config, src_voc, trg_voc, lexicon):
  in_size, out_size = len(src_voc), len(trg_voc)
  if network_config.model == "encdec":
    return nmtrain.models.nmt_encdec.EncoderDecoderNMT(
      hidden_units  = network_config.hidden_units,
      drop_out     = learning_config.dropout,
      lstm_depth   = network_config.depth,
      in_size      = in_size,
      out_size     = out_size
    )
  elif network_config.model == "attn":
    return nmtrain.models.nmt_attentional.AttentionalNMT(
      hidden_units    = network_config.hidden_units,
      drop_out       = learning_config.dropout,
      lstm_depth     = network_config.depth,
      in_size        = in_size,
      out_size       = out_size,
      input_feeding  = network_config.input_feeding,
      attention_type = network_config.attention_type,
      # (Arthur et al., 2016)
      lexicon        = lexicon,
    )
  else:
    raise Exception("Unknown Model Type:", model_type)

