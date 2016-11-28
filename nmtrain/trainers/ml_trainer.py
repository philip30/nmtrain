import numpy

import nmtrain
import nmtrain.data
import nmtrain.model
import nmtrain.serializer
import nmtrain.watcher
import nmtrain.log as log

class MaximumLikelihoodTrainer:
  def __init__(self, args):
    self.nmtrain_model = nmtrain.NmtrainModel(args)
    self.data_manager  = nmtrain.data.DataManager()
    # Training Parameters
    self.maximum_epoch = args.epoch
    self.bptt_len = args.bptt_len
    # Location of output model
    self.model_file = args.model_out
    # Load in the real data
    log.info("Loading Data")
    self.data_manager.load_train(args.src, args.trg,
                                 self.nmtrain_model.src_vocab,
                                 self.nmtrain_model.trg_vocab,
                                 args.src_dev, args.trg_dev,
                                 args.src_test, args.trg_test,
                                 args.batch, args.unk_cut)
    log.info("Loading Finished.")
    # Finalize the model, according to the data
    self.nmtrain_model.finalize_model(args)

  def train(self, classifier):
    xp      = nmtrain.environment.array_module()
    state   = self.nmtrain_model.training_state
    # Watcher is the supervisor of this training
    # It will watch and record everything happens during training
    watcher = nmtrain.watcher.TrainingWatcher(state,
                                              self.nmtrain_model.src_vocab,
                                              self.nmtrain_model.trg_vocab,
                                              self.data_manager.total_trg_words())
    # The original chainer model
    model   = self.nmtrain_model.chainer_model
    # Our data manager
    data    = self.data_manager
    # Chainer optimizer
    optimizer = self.nmtrain_model.optimizer

    # If test data is provided, prepare the appropriate watcher
    if data.has_test_data():
      self.test_state = nmtrain.model.TestState()
      test_watcher = nmtrain.watcher.TestWatcher(self.test_state,
                                                 self.nmtrain_model.src_vocab,
                                                 self.nmtrain_model.trg_vocab)

    def bptt(batch_loss):
      """ Backpropagation through time """
      model.zerograds()
      batch_loss.backward()
      batch_loss.unchain_backward()
      optimizer.update()

    # Before Training Describe the model
    self.nmtrain_model.describe()

    # Training with maximum likelihood estimation
    data.arrange(state.batch_indexes)
    for ep in range(state.finished_epoch, self.maximum_epoch):
      watcher.begin_epoch()
      for src_batch, trg_batch in data.train_data():
        # Convert to appropriate array
        if xp != numpy:
          src_data = xp.array(src_batch.data, dtype=numpy.int32)
          trg_data = xp.array(trg_batch.data, dtype=numpy.int32)
        else:
          src_data = src_batch.data
          trg_data = trg_batch.data
        # Prepare for training
        batch_loss = classifier.train(model, src_data, trg_data,
                                      watcher, bptt,
                                      bptt_len=self.bptt_len)
        watcher.batch_update(loss=batch_loss.data, size=len(trg_batch.data[0]))
        bptt(batch_loss)
      watcher.end_epoch(data.shuffle())

      # Evaluation on Development set
      if data.has_dev_data():
        nmtrain.environment.set_test()
        watcher.begin_evaluation()
        for src_sent, trg_sent in data.dev_data():
          if xp != numpy:
            src_data = xp.array(src_sent.data, dtype=numpy.int32)
            trg_data = xp.array(trg_sent.data, dtype=numpy.int32)
          else:
            src_data = src_sent.data
            trg_data = trg_sent.data
          # Prepare for evaluation
          classifier.test(model, src_data, watcher, trg_data=trg_data, force_limit=True)
        watcher.end_evaluation(*data.dev_batches)
        nmtrain.environment.set_train()

      # Incremental testing if wished
      if data.has_test_data():
        nmtrain.environment.set_test()
        test_watcher.begin_evaluation()
        for src_sent, trg_sent in data.test_data():
          if xp != numpy:
            src_data = xp.array(src_sent.data, dtype=numpy.int32)
            trg_data = xp.array(trg_sent.data, dtype=numpy.int32)
          else:
            src_data = src_sent.data
            trg_data = trg_sent.data
          classifier.test(model, src_data, test_watcher, trg_data=trg_data, force_limit=False)
        test_watcher.end_evaluation(*data.test_batches)
        nmtrain.environment.set_train()

      # Stop Early, otherwise, save
      if watcher.should_early_stop():
        if watcher.should_save():
          nmtrain.serializer.save(self.nmtrain_model, self.model_file)
        break
      else:
        nmtrain.serializer.save(self.nmtrain_model, self.model_file)

