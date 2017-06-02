import chainer
import numpy
import functools
import math
import itertools
import nmtrain

from chainer.functions import expand_dims
from chainer.functions import concat
from chainer.functions import broadcast_to
from chainer.functions import softmax, softmax_cross_entropy
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
    # Init debug
    nmtrain.debug.init(self.model.src_vocab, self.model.trg_vocab)

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
    self.samples = []
    ### Generating Negative Samples 
    for i, batch in enumerate(self.idomain_src.train_data):
      self.samples.append((batch.normal_data, False))
    ### Generating Positive Samples
    for i, batch in enumerate(self.idomain_trg.train_data):
      self.samples.append((batch.normal_data, True))

  def __call__(self, classifier):
    # Configuring Minimum Risk Training with discriminator
    learning_config = self.config.learning_config
    discriminator_loss = DiscriminatorLoss(self.discriminator, self.target_embedding)
    classifier.minrisk = nmtrain.minrisk.minrisk.MinimumRiskTraining(learning_config.learning.mrt,
                                                                     discriminator_loss)
    self.serializer   = TrainModelWriter(self.config.model_out, False)
    outputer = nmtrain.outputers.Outputer(self.model.src_vocab, self.model.trg_vocab)
    outputer.register_outputer("test", self.config.output_config.test)
    watcher = nmtrain.structs.watchers.Watcher(nmtrain.structs.nmtrain_state.NmtrainState())
    tester = nmtrain.testers.tester.Tester(watcher, classifier, self.model, outputer, self.config.test_config)

    #1. Pretrain the discriminator
    nmtrain.log.info("Pretraining discriminator for %d epochs" % learning_config.pretrain_epoch)
    self.train_discriminator(classifier, total_epoch=learning_config.pretrain_epoch)

    #2. Adversarial Training
    for i in range(learning_config.seqgan_epoch):
      self.train_generator(classifier, learning_config.generator_epoch, i)
      self.train_discriminator(classifier, learning_config.discriminator_epoch)

      if self.idomain_src.has_test_data:
        outputer.test.begin_collection(i)
        tester(model    = self.generator,
               data     = self.idomain_src.test_data,
               mode     = nmtrain.testers.TEST,
               outputer = outputer.test)
        outputer.test.end_collection()
      #3. Saving Model
      self.serializer.save(self.model)

      watcher.new_epoch()

  def train_generator(self, classifier, total_epoch, seqgan_epoch):
    self.generator.enable_update()
    self.discriminator.disable_update()
    if total_epoch == 0: return None
    total_loss = 0
    for epoch in range(total_epoch):
      self.minrisk_data.arrange(total_epoch * seqgan_epoch + epoch)
      epoch_loss = 0
      trained = 0
      for i, batch in enumerate(self.minrisk_data.train_data):
        src_batch, trg_batch = batch.normal_data
        try:
          loss = classifier.train_mrt(self.generator, src_batch, trg_batch, self.eos_id, self.model.trg_vocab) / len(batch)
          self.generator_bptt(loss)
        except:
          nmtrain.log.warning("Died at this batch_id:", batch.id, "with shape:", src_batch.shape, trg_batch.shape)
          raise

        trained += trg_batch.shape[1]
        nmtrain.log.info("[%d] Generator, Trained:%5d, loss=%5.3f" % (epoch+1, trained, loss.data))
        epoch_loss += loss
      epoch_loss /= i
      nmtrain.log.info("[%d] Generator Epoch summary: loss=%.3f" % (epoch+1, epoch_loss.data))

  def train_discriminator(self, classifier, total_epoch):
    self.discriminator.enable_update()
    self.generator.disable_update()
    if total_epoch == 0: return None
    gen_limit = self.config.learning_config.learning.mrt.generation_limit
    samples = self.samples

    # Produce an embedding of size (B, E, |F|) and 
    # a vector label of size B.
    def to_embed_vector(sample, is_positive):
      src_batch, trg_batch = sample
      trg_batch = self.generator.xp.array(trg_batch, dtype=numpy.int32)
      label = 1 if is_positive else 0
      if is_positive:
        trg_embedding = [expand_dims(self.target_embedding(word), axis=2) for word in trg_batch]
      else:
        with chainer.using_config('train', False):
          with chainer.no_backprop_mode():
            trg_embedding = classifier.generate(self.generator,
                                                src_batch,
                                                self.eos_id,
                                                gen_limit=gen_limit)
      trg_embedding = concat(trg_embedding, axis=2)
      # Ground Truth label
      ground_truth = self.create_label(trg_embedding.shape[0], label)
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
      for sample, is_positive in samples:
        embed, ground_truth = to_embed_vector(sample, is_positive)
        embed = expand_dims(embed, axis=1)
        # Discriminate the target
        output = self.discriminator(embed)
        # Calculate Loss
        loss = softmax_cross_entropy(output, ground_truth) / embed.shape[0]
        self.discriminator_bptt(loss)
        trained += embed.shape[0]
        nmtrain.log.info("[%d] Discriminator, Trained: %5d, loss=%5.3f" % (epoch + 1, trained, loss.data))
        epoch_loss += loss
      epoch_loss /= len(samples)
      total_loss += epoch_loss.data
      nmtrain.log.info("[%d] Discriminator Epoch Summary: loss=%.3f" % (epoch+1, epoch_loss.data))

    return total_loss / total_epoch

  @functools.lru_cache(maxsize=2)
  def create_label(self, size, label):
    return self.model.seqgan_model.xp.array([label for _ in range(size)], dtype= numpy.int32)

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
  def __init__(self, discriminator, target_embed):
    self.discriminator = discriminator
    self.target_embed = target_embed

  def __call__(self, sample):
    with chainer.using_config('train', False):
      with chainer.no_backprop_mode():
        return self.calculate_loss(sample)

  def calculate_loss(self, sample):
    sample = self.discriminator.xp.array(sample, dtype=numpy.int32)
    trg_embedding = [expand_dims(self.target_embed(word), axis=2) for word in sample.transpose()]
    trg_embedding = expand_dims(concat(trg_embedding, axis=2), axis=1)
    # Discriminator
    prob = self.discriminator(trg_embedding)
    prob = softmax(prob)
    prob.to_cpu()
    return -prob.data.transpose()[1]

