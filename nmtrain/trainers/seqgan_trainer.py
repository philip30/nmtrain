import chainer
import numpy
import functools
import math
import itertools
import nmtrain

from chainer.functions import expand_dims
from chainer.functions import concat
from chainer.functions import broadcast_to
from nmtrain.serializers import TrainModelWriter

class SequenceGANTrainer(object):
  def __init__(self, config):
    self.model         = nmtrain.structs.seqgan_model.NmtrainSeqGANModel(config)
    self.idomain_src  = nmtrain.data.data_manager.DataManager()
    self.idomain_trg  = nmtrain.data.data_manager.DataManager()
    self.minrisk_data = nmtrain.data.data_manager.DataManager()
    self.generator     = self.model.chainer_model
    self.discriminator = self.model.seqgan_model
    self.eos_id        = self.model.trg_vocab.eos_id()
    self.config        = self.model.seqgan_config
    # Vocabularies
    self.model.src_vocab.set_frozen(True)
    self.model.trg_vocab.set_frozen(True)

    # Note:
    # The vocabulary has been frozen here
    nmtrain.log.info("Loading data...")
    self.idomain_src.load_train(config.corpus.in_domain_src, config.data_config, self.model)
    self.idomain_trg.load_train(config.corpus.in_domain_trg, config.data_config, self.model)
    self.minrisk_data.load_train(config.corpus.in_domain_src, config.mrt_data_config, self.model)
    nmtrain.log.info("Loading finished.")

    # Reference to embedding function of the generator
    self.source_embedding = self.model.chainer_model.encoder.embed
    self.target_embedding = self.model.chainer_model.decoder.output_embed

    # Describe the model after finished loading
    self.model.finalize_model()
    nmtrain.log.info("SRC VOCAB:", len(self.model.src_vocab))
    nmtrain.log.info("TRG VOCAB:", len(self.model.trg_vocab))

    # Generate Positive + negative examples
    positive_samples = []
    negative_samples = []
    ### Generating Negative Samples 
    for i, batch in enumerate(self.idomain_src.train_data):
      negative_samples.append(batch.normal_data)
    ### Generating Positive Samples
    for i, batch in enumerate(self.idomain_trg.train_data):
      positive_samples.append(batch.normal_data)
    self.samples = []
    for pos_s, neg_s in itertools.zip_longest(positive_samples, negative_samples, fillvalue=None):
      self.samples.append((pos_s, neg_s))

  def __call__(self, classifier):
    # Configuring Minimum Risk Training with discriminator
    learning_config = self.config.learning_config
    discriminator_loss = DiscriminatorLoss(self.discriminator, self.target_embedding,
                                           self.eos_embed(), self.config.learning_config.generation_limit)
    classifier.minrisk = nmtrain.minrisk.minrisk.MinimumRiskTraining(learning_config.learning.mrt,
                                                                     discriminator_loss)
    self.serializer   = TrainModelWriter(self.config.model_out, False)

    #1. Pretrain the discriminator
    nmtrain.log.info("Pretraining discriminator for %d epochs" % learning_config.pretrain_epoch)
    self.train_discriminator(classifier, total_epoch=learning_config.pretrain_epoch)

    #2. Adversarial Training
    for _ in range(learning_config.seqgan_epoch):
      self.train_generator(classifier, 1)
      self.train_discriminator(classifier, 1)

      #3. Saving Model
      self.serializer.save(self.model)

  def train_generator(self, classifier, total_epoch):
    if total_epoch == 0: return None
    classifier.set_train(True)
    total_loss = 0
    for epoch in range(total_epoch):
      epoch_loss = 0
      trained = 0
      for batch in self.minrisk_data.train_data:
        src_batch, trg_batch = batch.normal_data
        loss = classifier.train_mrt(self.generator, src_batch, trg_batch, self.eos_id) / len(batch)
        self.generator_bptt(loss)
        trained += trg_batch.shape[1]
        nmtrain.log.info("[%d] Generator, Trained:%5d, loss=%5.3f" % (epoch+1, trained, loss.data))
        epoch_loss += loss
      epoch_loss /= len(self.minrisk_data.train_data)
      nmtrain.log.info("[%d] Generator Epoch summary: loss=%.3f" % (epoch+1, epoch_loss.data))

  def train_discriminator(self, classifier, total_epoch):
    if total_epoch == 0: return None
    classifier.set_train(False)
    gen_limit = self.config.learning_config.generation_limit
    samples = self.samples

    # Produce an embedding of size (B, E, |F|) and 
    # a vector label of size B.
    def to_embed_vector(sample, is_negative):
      src_batch, trg_batch = sample
      label = 0 if is_negative else 1
      if is_negative:
        trg_embedding = classifier.generate(self.generator,
                                             src_batch,
                                             self.eos_id,
                                             gen_limit=gen_limit)
      else:
        trg_embedding = [expand_dims(self.target_embedding(word), axis=2) for word in trg_batch]
      trg_embedding = concat(trg_embedding, axis=2)
      # padding
      shape = trg_embedding.shape
      pad = broadcast_to(self.eos_embed(), (shape[0], shape[1], gen_limit-shape[2]))
      trg_embedding = concat((trg_embedding, pad), axis=2)
      trg_embedding = expand_dims(trg_embedding, axis=1)
      # Ground Truth label
      batch_size = trg_embedding.shape[0]
      ground_truth = self.create_label(batch_size, label)
      return trg_embedding, ground_truth

    # Begin Discriminator Training
    total_loss = 0
    for epoch in range(total_epoch):
      numpy.random.shuffle(samples)
      epoch_loss = 0
      trained = 0
      # Train the positive sample and negative sample at once.
      # Putting them on the same batch together make the system more
      # adapt to distinguish between which one is positive and which one is negative.
      for pos_s, neg_s in samples:
        trg_embeddings = []
        labels = []
        if pos_s is not None:
          embed, gt = to_embed_vector(pos_s, False)
          trg_embeddings.append(embed)
          labels.append(gt)
        if neg_s is not None:
          embed, gt = to_embed_vector(neg_s, True)
          trg_embeddings.append(embed)
          labels.append(gt)
        trg_embedding = concat(trg_embeddings, axis=0)
        ground_truth = concat(labels, axis=0)
        # Discriminate the target
        output = self.discriminator(trg_embedding)
        # Calculate Loss
        loss = chainer.functions.softmax_cross_entropy(output, ground_truth) / trg_embedding.shape[0]
        self.discriminator_bptt(loss)
        trained += trg_embedding.shape[0]
        nmtrain.log.info("[%d] Discriminator, Trained: %5d, loss=%5.3f" % (epoch + 1, trained, loss.data)) 
        epoch_loss += loss
      epoch_loss /= len(samples)
      total_loss += epoch_loss.data
      nmtrain.log.info("[%d] Discriminator Epoch Summary: loss=%.3f" % (epoch+1, epoch_loss.data))

    return total_loss / total_epoch

  @functools.lru_cache(maxsize=2)
  def create_label(self, size, label):
    return self.model.seqgan_model.xp.array([label for _ in range(size)], dtype= numpy.int32)

  @functools.lru_cache(maxsize=1)
  def eos_embed(self):
    eos_embed = self.target_embedding(
                chainer.Variable(self.model.seqgan_model.xp.array([self.model.trg_vocab.eos_id()], dtype=numpy.int32)))
    eos_embed = chainer.functions.expand_dims(eos_embed, axis=2)
    eos_embed.volatile = chainer.ON
    return eos_embed

  def generator_bptt(self, loss):
    self.generator.cleargrads()
    loss.backward()
    loss.unchain_backward()
    self.model.gen_opt.update()

  def discriminator_bptt(self, loss):
    self.discriminator.cleargrads()
    loss.backward()
    loss.unchain_backward()
    self.model.dis_opt.update()

class DiscriminatorLoss(object):
  def __init__(self, discriminator, target_embed, eos_embed, gen_limit):
    self.discriminator = discriminator
    self.target_embed = target_embed
    self.eos_embed = eos_embed
    self.gen_limit = gen_limit

  def __call__(self, sample):
    trg_embedding = [expand_dims(self.target_embed(word), axis=2) for word in sample]
    trg_embedding = concat(trg_embedding, axis=2)
    # padding
    shape = trg_embedding.shape
    pad = broadcast_to(self.eos_embed, (shape[0], shape[1], self.gen_limit-shape[2]))
    trg_embedding = concat((trg_embedding, pad), axis=2)
    trg_embedding = expand_dims(trg_embedding, axis=1)
    prob = self.discriminator(trg_embedding, is_train=False)
    prob = chainer.functions.softmax(prob).data
    return -prob.transpose()[1]

