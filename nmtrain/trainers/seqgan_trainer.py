import chainer
import numpy
import functools
import math

import nmtrain

class SequenceGANTrainer(object):
  def __init__(self, args):
    self.seqgan_model  = nmtrain.NmtrainSeqGANModel(args)
    specification      = self.seqgan_model.specification
    self.orig_dm       = nmtrain.data.DataManager()
    self.dest_dm       = nmtrain.data.DataManager()
    self.generator     = self.seqgan_model.chainer_model
    self.discriminator = self.seqgan_model.discriminator_model
    self.eos_id        = self.seqgan_model.trg_vocab.eos_id()
    self.gen_limit     = args.gen_limit
    self.adapt_epoch   = args.adapt_epoch
    self.model_out     = args.model_out
    # Loading data
    src_vocab = self.seqgan_model.src_vocab
    trg_vocab = self.seqgan_model.trg_vocab
    src_vocab.set_frozen(True)
    trg_vocab.set_frozen(True)

    # Note:
    # 1. The vocabulary has been frozen here
    self.orig_dm.load_train(src             = specification.orig_src,
                            trg             = specification.orig_trg,
                            src_voc         = src_vocab,
                            trg_voc         = trg_vocab,
                            batch_size      = specification.batch,
                            sort_method     = specification.sort_method,
                            max_sent_length = specification.max_sent_length,
                            batch_strategy  = specification.batch_strategy,
                            bpe_codec       = self.seqgan_model.bpe_codec)

    self.dest_dm.load_train(src             = specification.dest_src,
                            trg             = specification.dest_trg,
                            src_voc         = src_vocab,
                            trg_voc         = trg_vocab,
                            batch_size      = specification.batch,
                            sort_method     = specification.sort_method,
                            max_sent_length = specification.max_sent_length,
                            batch_strategy  = specification.batch_strategy,
                            bpe_codec       = self.seqgan_model.bpe_codec)

    # Reference to embedding function of the generator
    self.source_embedding = self.seqgan_model.chainer_model.encoder.embed
    self.target_embedding = self.seqgan_model.chainer_model.decoder.output_embed

    # Describe the model after finished loading
    self.seqgan_model.describe()

  def train(self, classifier):
    #1. Pretrain the discriminator
    nmtrain.log.info("Pretraining discriminator for 10 epochs")
    self.train_discriminator(classifier, epoch=3)

    #2. Adversarial Training
    for _ in range(self.adapt_epoch):
      self.train_generator(classifier, 1)
      self.train_discriminator(classifier, 1)

    #3. Saving Model
    nmtrain.serializer.save(self.seqgan_model, self.model_out)

  def train_generator(self, classifier, epoch):
    xp  = self.seqgan_model.discriminator_model.xp
    for ep in range(epoch):
      epoch_loss = 0
      for i, batch in enumerate(self.orig_dm.train_data):
        src_batch, trg_batch = batch.normal_data
        trg_embedding = classifier.generate(self.generator, src_batch, self.eos_id)
        # Discriminator
        nmtrain.environment.set_test()
        output = self.discriminator(trg_embedding)
        nmtrain.environment.set_train()
        # Counterfeiting
        batch_size   = trg_embedding[0].shape[0]
        ground_truth = self.create_label(xp, batch_size, 1)
        # Calculate Loss
        loss = chainer.functions.softmax_cross_entropy(output, ground_truth)
        self.generator_bptt(loss)
        epoch_loss += loss
      epoch_loss /= i
      nmtrain.log.info("Generator train epoch#%d: PPL=%.3f" % (ep+1, math.exp(epoch_loss.data)))

  def train_discriminator(self, classifier, epoch):
    xp  = self.seqgan_model.discriminator_model.xp

    samples = []
    ### Generating Negative Samples 
    for i, batch in enumerate(self.orig_dm.train_data):
      src_batch, trg_batch = batch.normal_data
      samples.append((src_batch, trg_batch, 0))
    ### Generating Positive Samples
    for i, batch in enumerate(self.dest_dm.train_data):
      src_batch, trg_batch = batch.normal_data
      samples.append((src_batch, trg_batch, 1))

    # Begin Discriminator Training
    loss = 0
    for ep in range(epoch):
      numpy.random.shuffle(samples)
      epoch_loss = 0
      for src_batch, trg_batch, label in samples:
        # Generate Embedding according to label
        nmtrain.environment.set_test()
        if label == 0:
          trg_embedding = classifier.generate(self.generator, src_batch, self.eos_id)
        else:
          trg_embedding = [self.target_embedding(trg) for trg in trg_batch]
        nmtrain.environment.set_train()
        # Ground Truth label
        batch_size = trg_embedding[0].shape[0]
        ground_truth = self.create_label(xp, batch_size, label)
        # Discriminate the target
        output = self.discriminator(trg_embedding)
        # Calculate Loss
        loss = chainer.functions.softmax_cross_entropy(output, ground_truth)
        self.discriminator_bptt(loss)

        epoch_loss += loss
      epoch_loss /= len(samples)
      nmtrain.log.info("Discriminator train epoch#%d: PPL=%.3f" % (ep+1, math.exp(epoch_loss.data)))

    return loss

  def create_label(self, xp, size, label):
    return xp.array([label for _ in range(size)], dtype= numpy.int32)

  def generator_bptt(self, loss):
    self.generator.cleargrads()
    loss.backward()
    loss.unchain_backward()
    self.seqgan_model.optimizer.update()

  def discriminator_bptt(self, loss):
    self.discriminator.cleargrads()
    loss.backward()
    loss.unchain_backward()
    self.seqgan_model.seqgan_optimizer.update()

