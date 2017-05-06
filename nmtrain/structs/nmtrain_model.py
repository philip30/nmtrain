import chainer
import nmtrain
import os

class NmtrainModel(object):
  def __init__(self, config):
    self.config = config
    if len(config.init_model) != 0 and os.path.exists(config.init_model):
      model_loader = nmtrain.serializers.TrainModelReader(self)
      model_loader.load(config)
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
    if self.lexicon is None and self.config.lexicon_config.path:
      nmtrain.log.info("Constructing lexicon...")
      self.lexicon = nmtrain.structs.lexicon.Lexicon(self.src_vocab, self.trg_vocab,
                                                     self.config.lexicon_config.alpha,
                                                     self.config.lexicon_config.method,
                                                     self.config.lexicon_config.path)
    else:
      self.lexicon = None

    if not hasattr(self, "bpe_codec"):
      self.bpe_codec = None, None

    if self.chainer_model is None:
      nmtrain.log.info("Constructing model...")
      self.chainer_model = from_spec(self.config.network_config, self.config.learning_config,
                                     self.src_vocab, self.trg_vocab, self.lexicon)
      nmtrain.log.info("Setting up optimizer...")
      self.optimizer.setup(self.chainer_model)

#      # Initializing model
#      nmtrain.log.info("Initializing weight uniformly [-0.1, 0.1]...")
#      initializer = chainer.initializers.Uniform(scale=0.1)
#      for name, array in sorted(self.chainer_model.namedparams()):
#        initializer(array.data)

    if hasattr(self, "optimizer"):
      self.optimizer.add_hook(chainer.optimizer.GradientClipping(self.config.learning_config.gradient_clipping))

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      nmtrain.log.info("Copying model to GPU")
      self.chainer_model.to_gpu(nmtrain.environment.gpu)
    nmtrain.log.info("Model constructed!")

  def describe(self):
    pass

  @property
  def xp(self):
    return self.chainer_model.xp

  @property
  def nmtrain_state(self):
    return self.state

## ROUTINES
def parse_optimizer(optimizer):
  opt = optimizer.type
  param = optimizer.params
  # Select optimizer
  if opt == "adam":
    return chainer.optimizers.Adam(alpha=float(param.adam.alpha),
                                   beta1=float(param.adam.beta1),
                                   beta2=float(param.adam.beta2),
                                   eps=float(param.adam.eps))
  elif opt == "sgd":
    return chainer.optimizers.SGD(lr=float(param.adam.lr))
  else:
    raise ValueError("Unrecognized optimizer:", opt)

def load_bpe_codec(config):
  if hasattr(config, "bpe_config") and \
      config.bpe_config.src_codec and \
      config.bpe_config.trg_codec:
    return  nmtrain.third_party.bpe.BPE(config.bpe_config.src_codec), \
            nmtrain.third_party.bpe.BPE(config.bpe_config.trg_codec)
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

