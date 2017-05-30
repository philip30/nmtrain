import chainer

import nmtrain
import nmtrain.models.discriminators as discriminators

from .nmtrain_model import NmtrainModel
from .nmtrain_model import parse_optimizer

class NmtrainSeqGANModel(NmtrainModel):
  def __init__(self, config):
    super(NmtrainSeqGANModel, self).__init__(config)
    self.seqgan_config = config
    if config.init_seqgan_model:
      # TODO(philip30): Added pretrained seqgan model here?
      pass
    else:
      self.seqgan_model = from_spec(self.config, config.discriminator, config.learning_config)
      self.gen_opt = parse_optimizer(config.learning_config.optimizer)
      self.dis_opt = parse_optimizer(config.learning_config.optimizer)
      self.gen_opt.setup(self.chainer_model)
      self.dis_opt.setup(self.seqgan_model)

    # SeqGAN optimizer
    self.gen_opt.use_cleargrads()
    self.dis_opt.use_cleargrads()
    self.gen_opt.add_hook(chainer.optimizer.GradientClipping(self.config.learning_config.gradient_clipping))
    self.dis_opt.add_hook(chainer.optimizer.GradientClipping(self.config.learning_config.gradient_clipping))

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      nmtrain.log.info("Copying model to GPU")
      self.seqgan_model.to_gpu(nmtrain.environment.gpu)
    nmtrain.log.info("Seqgan model constructed!")

  def describe(self):
    super(NmtrainSeqGANModel, self).describe()

def from_spec(nmt_config, network_config, learning_config):
  embed = nmt_config.network_config.hidden_units.embed
  generation_limit = learning_config.generation_limit
  return discriminators.Conv2DTargetDiscriminator(network_config.hidden_units, learning_config.dropout, embed, generation_limit)

