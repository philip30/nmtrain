import chainer
import numpy

import nmtrain

class SequenceGANTrainer(object):
  def __init__(self, args):
    self.seqgan_model = nmtrain.NmtrainSeqGANModel(args)
    self.orig_dm      = nmtrain.data.DataManager()
    self.dest_dm      = nmtrain.data.DataManager()
    self.generator    = self.seqgan_model.chainer_model
    self.discriminator = self.seqgan_model.discriminator_model
    self.batch_size   = args.batch
    # Loading data
    src_vocab = self.seqgan_model.src_vocab
    trg_vocab = self.seqgan_model.trg_vocab
    # Note the max sent length is set to 100 here.`
    self.orig_dm.load_test(src=args.orig_src, ref=args.orig_trg, src_voc=src_vocab, trg_voc=trg_vocab, batch_size=args.batch, sort=True)
    self.dest_dm.load_test(src=args.dest_src, ref=args.dest_trg, src_voc=src_vocab, trg_voc=trg_vocab, batch_size=args.batch, sort=True)

    # Reference to embedding function
    self.source_embedding = self.seqgan_model.chainer_model.encoder.embed
    self.target_embedding = self.seqgan_model.chainer_model.decoder.output_embed

    # Describe the model after finished loading
    self.seqgan_model.describe()

  def train(self, classifier):
    #1. Pretrain the discriminator
    negative_samples = self.generate_negative_samples(classifier)
    positive_samples = self.dest_dm.test_data
    print(self.train_discriminator(positive_samples, negative_samples))

  def train_discriminator(self, positive_samples, negative_samples, epoch=10):
    discriminator = self.seqgan_model.discriminator_model
    optimizer = self.seqgan_model.seqgan_optimizer

    # gt = ground truth
    positive_gt = create_batch_label(discriminator.xp, 1, self.batch_size)
    negative_gt = create_batch_label(discriminator.xp, 0, self.batch_size)

    def bptt(loss):
      loss.backward()
      loss.unchain_backward()
      optimizer.update()

    # Begin Discriminator Training
    discriminator.cleargrads()
    for ep in range(epoch):
      epoch_loss = 0
      # Negative Example
      for ngt_ctr, (src_batch, trg_batch) in enumerate(negative_samples):
        score = discriminator.discriminate_target(trg_batch, self.target_embedding)
        if trg_batch.shape[1] != negative_gt.shape[0]:
          true_score = create_batch_label(discriminator.xp, 0, negative_gt.shape[0])
        else:
          true_score = negative_gt
        loss = chainer.functions.mean_squared_error(score, true_score)
        epoch_loss += float(loss.data)
        bptt(loss)
      # Positive Example
      for pst_ctr, (src_batch, trg_batch) in enumerate(positive_samples):
        trg_batch = trg_batch.data
        score = discriminator.discriminate_target(trg_batch, self.target_embedding)
        if trg_batch.shape[1] != positive_gt.shape[0]:
          true_score = create_batch_label(discriminator.xp, 1, positive_gt.shape[0])
        else:
          true_score = positive_gt
        loss = chainer.functions.mean_squared_error(score, true_score)
        epoch_loss += float(loss.data)
        bptt(loss)
      epoch_loss /= (ngt_ctr + pst_ctr)
    return epoch_loss

  def generate_negative_samples(self, classifier):
    nmtrain.environment.set_test()
    negative_batches = []
    for src_batch, trg_batch in self.orig_dm.test_data:
      negative_batches.append((src_batch.data,
                               classifier.generate(self.generator,
                                                   src_batch,
                                                   self.seqgan_model.trg_vocab.eos_id())))
    nmtrain.environment.set_train()
    return negative_batches

def create_batch_label(xp, label, size):
  return xp.array([[label] for _ in range(size)], dtype=numpy.float32)
