import numpy
import gc
import math

import nmtrain

class NMTTrainer:
  def __init__(self, config):
    # NmtrainModel encapsulates the chainer model.
    self.nmtrain_model = nmtrain.NmtrainModel(config)
    # Data Manager take care of the data
    self.data_manager  = nmtrain.data.DataManager()
    # Unknown Trainer for unknown word
    self.unknown_trainer = nmtrain.trainers.unknown_trainers.from_config(config.data_config.unknown_training)
    # Training Parameters
    corpus          = config.corpus
    # Load in the real data
    nmtrain.log.info("Loading Data")
    self.data_manager.load_train(config.corpus, config.data_config, self.nmtrain_model)
    nmtrain.log.info("Loading Finished.")
    # Finalize the model, according to the data
    self.nmtrain_model.finalize_model()

  def __call__(self, classifier):
    state           = self.nmtrain_model.state
    learning_config = self.nmtrain_model.config.learning_config
    output_config   = self.nmtrain_model.config.output_config
    test_config     = self.nmtrain_model.config.test_config
    # The outputers is the one who is responsible in outputing the 
    # results to screen, or other streams
    outputer = nmtrain.outputers.Outputer(self.nmtrain_model.src_vocab, self.nmtrain_model.trg_vocab)
    outputer.register_outputer("train", output_config.train)
    outputer.register_outputer("dev", output_config.dev)
    outputer.register_outputer("test", output_config.test)
    # Updater
    watcher = nmtrain.structs.watchers.Watcher(state)
    # Tester
    tester = nmtrain.testers.tester.Tester(watcher, classifier, self.nmtrain_model, outputer, test_config)
    # The original chainer model
    model   = self.nmtrain_model.chainer_model
    # Our data manager
    data    = self.data_manager
    # Chainer optimizer
    optimizer = self.nmtrain_model.optimizer

    # Special case for word_dropout unknown trainer
    if self.unknown_trainer.__class__ == nmtrain.trainers.unknown_trainers.UnknownWordDropoutTrainer:
      self.unknown_trainer.src_freq_map = data.analyzer.src_analyzer.count_word_id_map(self.nmtrain_model.src_vocab)
      self.unknown_trainer.trg_freq_map = data.analyzer.trg_analyzer.count_word_id_map(self.nmtrain_model.trg_vocab)

    # BPTT callback
    def bptt(batch_loss):
      """ Backpropagation through time """
      if math.isnan(float(batch_loss.data)):
        nmtrain.log.warning("Loss is NaN, skipping update.")
        return

      model.cleargrads()
      batch_loss.backward()
      batch_loss.unchain_backward()
      optimizer.update()

    # Configure classfier
    classifier.configure_learning(bptt, learning_config)

    # Before Training Describe the model
    nmtrain.log.info("\n", str(self.nmtrain_model.config))

    # Training with maximum likelihood estimation
    start_epoch = state.last_trained_epoch
    end_epoch   = learning_config.epoch
    self.nmtrain_model.state.record_start_epoch(self.nmtrain_model.config)
    for ep in range(start_epoch, end_epoch):
      ep_arrangement = data.arrange(ep)

      # Training Iterations
      watcher.begin_train_epoch()
      model.set_train(True)
      classifier.set_train(True)
      for batch in data.train_data:
        for batch_retriever in self.unknown_trainer:
          src_batch, trg_batch = batch_retriever(batch)

          watcher.begin_batch()
          # Prepare for training
          batch_loss = classifier.train(model, src_batch, trg_batch, outputer.train)
          # BPTT 
          bptt(batch_loss)
          # Generate summary of batch training and keep track of it
          watcher.end_batch(loss=batch_loss.data,
                            src_shape=src_batch.shape,
                            trg_shape=trg_batch.shape,
                            batch_id=batch.id)
      watcher.end_train_epoch()

      # Cleaning up
      gc.collect()

      # Evaluation on Development set
      if data.has_dev_data:
        outputer.dev.begin_collection(ep)
        tester(model    = model,
               data     = data.dev_data,
               mode     = nmtrain.testers.DEV,
               outputer = outputer.dev)
        outputer.dev.end_collection()

      # Incremental testing if wished
      if data.has_test_data:
        outputer.test.begin_collection(ep)
        tester(model    = model,
               data     = data.test_data,
               mode     = nmtrain.testers.TEST,
               outputer = outputer.test)
        outputer.test.end_collection()

      # Check if dev perplexities decline
      dev_ppl_decline = state.is_ppl_decline()

      # LR Decay
      if ep + 1 >= learning_config.lr_decay.after_iteration or dev_ppl_decline:
        if optimizer.__class__.__name__ == "SGD":
          optimizer.lr *= learning_config.lr_decay.factor
          nmtrain.log.info("Decreasing lr by %f, now sgd lr: %f" % (learning_config.lr_decay.factor, optimizer.lr))
        elif optimizer.__class__.__name__ == "Adam":
          optimizer.alpha *= learning_config.lr_decay.factor
          nmtrain.log.info("Decreasing alpha by %f, now adam lr: %f", (learning_config.lr_decay.factor, optimizer.alpha))

      # Tell Chainer optimizer to increment the epoch
      optimizer.new_epoch()
      state.new_epoch()

      # Save the model
      outputer.train.save_model(self.nmtrain_model)

      # Early stopping
      ppl_worse_counter = learning_config.early_stop.ppl_worse_counter
      if ppl_worse_counter != 0 and state.no_dev_ppl_improvement_after(ppl_worse_counter):
        nmtrain.log.info("No dev ppl improvement after %d iterations. Finishing early." %(ppl_worse_counter))
        break

