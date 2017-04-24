import chainer

import nmtrain

class NmtrainSeqGANModel(nmtrain.NmtrainModel):
  def __init__(self, args):
    super(NmtrainSeqGANModel, self).__init__(args)
    if args.init_seqgan_model:
      # TODO(philip30): Added pretrained seqgan model here?
      pass
    else:
      self.discriminator_model = from_spec(self.specification)
      self.seqgan_optimizer = chainer.optimizers.Adam()
      self.seqgan_optimizer.setup(self.discriminator_model)

    self.seqgan_optimizer.use_cleargrads()
    self.seqgan_optimizer.add_hook(chainer.optimizer.GradientClipping(self.specification.gradient_clipping))

    # Put the model into GPU if used
    if nmtrain.environment.use_gpu():
      self.discriminator_model.to_gpu(nmtrain.environment.gpu)

  def describe(self):
    super(NmtrainSeqGANModel, self).describe()

def from_spec(specification):
  model = nmtrain.models.discriminators.RNNTargetDiscriminator(specification.embed,
                                                               specification.hidden,
                                                               specification.dropout)
  return model

