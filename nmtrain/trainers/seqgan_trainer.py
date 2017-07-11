import chainer
import numpy
import functools
import math
import itertools
import nmtrain

from chainer.functions import expand_dims
from chainer.functions import concat
from chainer.functions import broadcast_to, squeeze
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

    nmtrain.log.info("Setting number items in data_config to 1")
    self.discriminator_batch_size = config.data_config.batch
    config.data_config.batch = 1

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

    # EOS embed
    self.ngram = self.config.discriminator.hidden_units.ngram
    with chainer.no_backprop_mode():
      self.eos_embed = expand_dims(expand_dims(self.target_embedding(self.discriminator.xp.asarray([self.eos_id])), axis=0), axis=3)

    ### Others
    discriminator_loss = DiscriminatorLoss(self.discriminator, self.target_embedding, self.pad)
    # Minrisk to train the generator
    self.minrisk = nmtrain.minrisk.minrisk.MinimumRiskTraining(config.learning_config.learning.mrt, discriminator_loss)
    # Serializer
    self.serializer = TrainModelWriter(self.config.model_out, False)
    # Outputer
    self.outputer = nmtrain.outputers.Outputer(self.model.src_vocab, self.model.trg_vocab)
    self.outputer.register_outputer("train", self.config.output_config.train)
    self.outputer.register_outputer("test", self.config.output_config.test)

  def __call__(self, classifier):
    # Configuring Minimum Risk Training with discriminator
    learning_config = self.config.learning_config
    classifier.minrisk = self.minrisk
    watcher = nmtrain.structs.watchers.Watcher(nmtrain.structs.nmtrain_state.NmtrainState())
    tester = nmtrain.testers.tester.Tester(watcher, classifier, self.model, self.outputer, self.config.test_config)


    if self.config.pretest:
      self.outputer.test.begin_collection(0)
      tester(model = self.generator, data=self.idomain_src.test_data, mode= nmtrain.testers.TEST, outputer = self.outputer.test)
      self.outputer.test.end_collection()

    #1. Pretrain the discriminator
    nmtrain.log.info("Pretraining discriminator for %d epochs" % learning_config.pretrain_epoch)
    self.train_discriminator(classifier, total_epoch=learning_config.pretrain_epoch)

    #2. Adversarial Training
    for i in range(learning_config.seqgan_epoch):
      self.train_generator(classifier, learning_config.generator_epoch, i)
      self.train_discriminator(classifier, learning_config.discriminator_epoch, i)

      if self.idomain_src.has_test_data:
        self.outputer.test.begin_collection(i+1)
        tester(model    = self.generator,
               data     = self.idomain_src.test_data,
               mode     = nmtrain.testers.TEST,
               outputer = self.outputer.test)
        self.outputer.test.end_collection()
      #3. Saving Model
      self.serializer.save(self.model)
      watcher.new_epoch()

  def train_generator(self, classifier, total_epoch, seqgan_epoch=0):
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
          loss = classifier.train_mrt(self.generator, src_batch, None,
                                      self.eos_id, self.outputer.train) / len(batch)
          self.generator_bptt(loss)
        except:
          nmtrain.log.warning("Died at this batch_id:", batch.id, "with shape:", src_batch.shape, trg_batch.shape)
          if self.config.hack_config.skip_training_exception:
            continue
          else:
            raise

        trained += trg_batch.shape[1]
        nmtrain.log.info("[%d] Generator, Trained:%5d, loss=%5.3f" % (epoch+1+seqgan_epoch, trained, loss.data))
        epoch_loss += loss
      epoch_loss /= i
      nmtrain.log.info("[%d] Generator Epoch summary: loss=%.3f" % (epoch+1+seqgan_epoch, epoch_loss.data))

  def train_discriminator(self, classifier, total_epoch, seqgan_epoch=0):
    if total_epoch == 0:
      return None

    nmtrain.log.info("Updating Discriminator, disable update for Generator")
    self.discriminator.enable_update()
    self.generator.disable_update()

    # Data point to hold embeddings, label, and word
    class DataPoint:
      def __init__(self, embed, label, word):
        self.embed = embed
        self.label = label
        self.word = word

      def __iter__(self):
        return iter([self.embed, self.label, self.word])

      def __len__(self):
        return self.embed.shape[1]

    # Generating Embeddings Samples
    with chainer.using_config('train', False):
      with chainer.no_backprop_mode():
        nmtrain.log.info("Generating Embeddings for both positive and negative samples")
        embeddings = []

        def list_to_embed(words):
          lst = concat(list(map(lambda word: self.target_embedding(self.generator.xp.asarray(word)), trg_batch)), axis=0).transpose()
          lst.to_cpu()
          return lst

        for batch in self.idomain_src.train_data:
          src_batch, trg_batch = batch.normal_data
          predict_output = classifier.predict(self.generator, src_batch,
                                              eos_id = self.eos_id,
                                              gen_limit = self.config.test_config.generation_limit,
                                              beam = self.config.test_config.beam)
          embeddings.append(DataPoint(list_to_embed(predict_output.prediction), 0, predict_output.prediction))
        for batch in self.idomain_trg.train_data:
          src_batch, trg_batch = batch.normal_data
          embeddings.append(DataPoint(list_to_embed(trg_batch), 1, trg_batch))
        # Shuffle them
        numpy.random.shuffle(embeddings)
        embeddings = sorted(embeddings, key=lambda x: x.embed.shape[1])

        # Batch them together
        eos_embed = self.target_embedding(self.generator.xp.asarray([self.eos_id])).transpose()
        eos_embed.to_cpu()
        def discriminator_post_process(batch):
          max_length = max(item.embed.shape[1] for item in batch)
          embeds, labels, words = [], [], []
          for embed, label, word in batch:
            deficit = max_length - embed.shape[1]

            if deficit > 0:
              embed = concat((embed, broadcast_to(eos_embed, (embed.shape[0], deficit))), axis=1)

            embeds.append(expand_dims(embed, axis=0))
            labels.append(expand_dims(numpy.asarray([label], dtype=numpy.int32), axis=0))
            words.append(word)

          batch.data = (concat(embeds, axis=0).data, squeeze(concat(labels, axis=0), axis=1).data, words)

        # Batch Manager
        batch_manager = nmtrain.data.BatchManager(strategy=self.config.data_config.batch_strategy)
        batch_manager.load(embeddings,
                           n_items = self.discriminator_batch_size,
                           postprocessor = discriminator_post_process)

    # Begin Discriminator Training
    total_loss = 0
    for epoch in range(total_epoch):
      batch_manager.shuffle(numpy.random)
      epoch_loss = 0
      trained = 0
      # Train the positive sample and negative sample at once.
      # Putting them on the same batch together make the system more
      # adapt to distinguish between which one is positive and which one is negative.
      for batch in batch_manager:
        embed, ground_truth, words = batch.data
        self.outputer.train.begin_collection((words, ground_truth))
        embed = self.discriminator.xp.asarray(embed)
        ground_truth = self.discriminator.xp.array(ground_truth)
        embed = self.pad(expand_dims(embed, axis=1))
        # Discriminate the target
        try:
          output = self.discriminator(embed)
          self.outputer.train(nmtrain.data.Data(disc_out=output))
        except:
          nmtrain.log.warning("Died at batch with embed shape: ", embed.shape)
          raise
        
        # Calculate Loss
        loss = softmax_cross_entropy(output, ground_truth)
        self.discriminator_bptt(loss)
        trained += embed.shape[0]
        nmtrain.log.info("[%d] Discriminator, Trained: %5d, loss=%5.3f" % (epoch + 1 + seqgan_epoch, trained, loss.data))
        epoch_loss += loss / embed.shape[0]
        self.outputer.train.end_collection()
      total_loss += epoch_loss.data
      nmtrain.log.info("[%d] Discriminator Epoch Summary: loss=%.3f" % (epoch + 1 + seqgan_epoch, epoch_loss.data))

    return total_loss / total_epoch

  def pad(self, trg_embed):
    if self.ngram > trg_embed.shape[3]:
      with chainer.no_backprop_mode():
        pad = broadcast_to(self.eos_embed, (trg_embed.shape[0], trg_embed.shape[1], trg_embed.shape[2],
                                                    self.ngram - trg_embed.shape[3]))
        return concat((trg_embed, pad), axis=3)
    else:
      return trg_embed

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
  def __init__(self, discriminator, target_embed, pad_function):
    self.discriminator = discriminator
    self.target_embed = target_embed
    self.pad = pad_function

  def __call__(self, sample):
    with chainer.using_config('train', False):
      with chainer.no_backprop_mode():
        return self.calculate_loss(sample)

  def calculate_loss(self, sample):
    sample = self.discriminator.xp.asarray(sample, dtype=numpy.int32)
    trg_embedding = [expand_dims(self.target_embed(word), axis=2) for word in sample.transpose()]
    trg_embedding = expand_dims(concat(trg_embedding, axis=2), axis=1)
    # Discriminator
    prob = self.discriminator(self.pad(trg_embedding))
    prob = softmax(prob)
    prob.to_cpu()
    return -prob.data.transpose()[1]

