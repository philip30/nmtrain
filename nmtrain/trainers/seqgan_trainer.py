import chainer
import numpy
import functools
import math

import nmtrain

class SequenceGANTrainer(object):
  def __init__(self, args):
    self.seqgan_model  = nmtrain.NmtrainSeqGANModel(args)
    self.orig_dm       = nmtrain.data.DataManager()
    self.dest_dm       = nmtrain.data.DataManager()
    self.generator     = self.seqgan_model.chainer_model
    self.discriminator = self.seqgan_model.discriminator_model
    self.eos_id        = self.seqgan_model.trg_vocab.eos_id()
    self.gen_limit     = args.gen_limit
    # Loading data
    src_vocab = self.seqgan_model.src_vocab
    trg_vocab = self.seqgan_model.trg_vocab
    src_vocab.set_frozen(True)
    trg_vocab.set_frozen(True)

    # Note:
    # 1. The vocabulary has been frozen here
    self.orig_dm.load_train(src             = args.orig_src,
                            trg             = args.orig_trg,
                            src_voc         = src_vocab,
                            trg_voc         = trg_vocab,
                            batch_size      = args.batch,
                            sort_method     = args.sort_method,
                            max_sent_length = args.max_sent_length,
                            batch_strategy  = args.batch_strategy,
                            bpe_codec       = self.seqgan_model.bpe_codec)

    self.dest_dm.load_train(src             = args.dest_src,
                            trg             = args.dest_trg,
                            src_voc         = src_vocab,
                            trg_voc         = trg_vocab,
                            batch_size      = args.batch,
                            sort_method     = args.sort_method,
                            max_sent_length = args.max_sent_length,
                            batch_strategy  = args.batch_strategy,
                            bpe_codec       = self.seqgan_model.bpe_codec)

    # Reference to embedding function of the generator
    self.source_embedding = self.seqgan_model.chainer_model.encoder.embed
    self.target_embedding = self.seqgan_model.chainer_model.decoder.output_embed

    # Describe the model after finished loading
    self.seqgan_model.describe()

  def train(self, classifier):
    #1. Pretrain the discriminator
    nmtrain.log.info("Pretraining discriminator for 10 epochs")
    self.train_discriminator(classifier, epoch=10)

  def train_discriminator(self, classifier, epoch):
    discriminator = self.seqgan_model.discriminator_model
    optimizer     = self.seqgan_model.seqgan_optimizer
    xp            = self.seqgan_model.chainer_model.xp

    def bptt(loss):
      discriminator.cleargrads()
      loss.backward()
      loss.unchain_backward()
      optimizer.update()

    # Begin of Generating Samples
    samples = []
    nmtrain.environment.set_test()
    ### Generating Negative Samples 
    for i, batch in enumerate(self.orig_dm.train_data):
      src_batch, trg_batch = batch.normal_data
      samples.append((classifier.generate(self.generator, src_batch, self.eos_id), 1))
    ### Generating Positive Samples
    for i, batch in enumerate(self.orig_dm.train_data):
      src_batch, trg_batch = batch.normal_data
      embedding = []
      for trg in trg_batch:
        embed = self.target_embedding(trg)
        embed.to_cpu()
        embedding.append(embed)
      samples.append((embedding, 0))
    # End of Generating Samples
    nmtrain.environment.set_train()

    # Shuffle Sample
    numpy.random.shuffle(samples)

    # Begin Discriminator Training
    loss = 0
    for ep in range(epoch):
      epoch_loss = 0
      for trg_embedding, label in samples:
        batch_size = trg_embedding[0].shape[0]
        ground_truth = self.create_label(xp, batch_size, label)
        # Discriminate the target
        output = discriminator(trg_embedding)
        # Calculate Loss
        loss = chainer.functions.softmax_cross_entropy(output, ground_truth)
        bptt(loss)

        epoch_loss += loss
      epoch_loss /= len(samples)
      nmtrain.log.info("Discriminator train epoch#%d: PPL=%.3f" % (ep+1, math.exp(epoch_loss.data)))

    return loss

  def create_label(self, xp, size, label):
    return xp.array([label for _ in range(size)], dtype= numpy.int32)

