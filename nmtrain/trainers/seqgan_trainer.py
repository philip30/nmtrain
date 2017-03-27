import chainer
import numpy
import functools

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
    nmtrain.log.info("Pretraining discriminator")
    data_generator = list(self.generate_both_data(classifier))
    print(self.train_discriminator(data_generator))

  def train_discriminator(self, data_generator, epoch=10):
    discriminator = self.seqgan_model.discriminator_model
    optimizer = self.seqgan_model.seqgan_optimizer
    xp = self.seqgan_model.chainer_model.xp

    def bptt(loss):
      loss.backward()
      loss.unchain_backward()
      optimizer.update()

    # Begin Discriminator Training
    discriminator.cleargrads()
    loss = 0
    for ep in range(epoch):
      epoch_loss = 0
      for data_size, (src_batch, trg_batch, label) in enumerate(data_generator):
        batch_size = trg_batch.shape[1]
        ground_truth = self.create_label(xp, batch_size, label)
        # Discriminate the target
        output = discriminator(src_batch, trg_batch, self.target_embedding)
        # Calculate Loss
        loss = chainer.functions.softmax_cross_entropy(output, ground_truth)
        bptt(loss)

        epoch_loss += loss
      epoch_loss /= data_size
      print(epoch_loss.data)

    return loss

  def create_label(self, xp, size, label):
    return xp.array([label for _ in range(size)], dtype= numpy.int32)

  def generate_samples(self, data, classifier, label, volatile):
    for batch in data.train_data:
      if volatile:
        nmtrain.environment.set_test()
      src_batch, trg_batch = batch.normal_data
      trg_gen = classifier.generate(self.generator, src_batch, self.eos_id, generation_limit=self.gen_limit)
      nmtrain.environment.set_train()
      yield src_batch, trg_gen, label

  def generate_both_data(self, classifier, negative_volatile=True, positive_volatile=True):
    for item in self.generate_samples(self.orig_dm, classifier, 0, negative_volatile):
      yield item

    for batch in self.dest_dm.train_data:
      src_batch, trg_batch = batch.normal_data
      yield src_batch, trg_batch, 1

