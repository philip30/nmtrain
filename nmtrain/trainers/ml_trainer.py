import numpy

import nmtrain
import nmtrain.data
import nmtrain.model
import nmtrain.serializer
import nmtrain.watcher
import nmtrain.log as log

class MaximumLikelihoodTrainer:
  def __init__(self, args):
    # If init_model is provided, args will be overwritten
    # Inside the constructor of NmtrainModel using the 
    # previous training specification.
    self.nmtrain_model = nmtrain.NmtrainModel(args)
    self.data_manager  = nmtrain.data.DataManager()
    # Training Parameters
    self.maximum_epoch  = args.epoch
    self.bptt_len       = args.bptt_len
    self.early_stop_num = args.early_stop
    self.save_models    = args.save_models
    # Location of output model
    self.model_file = args.model_out
    # SGD lr decay factor
    self.sgd_lr_decay_factor = args.sgd_lr_decay_factor
    self.sgd_lr_decay_after  = args.sgd_lr_decay_after
    # Testing configuration
    self.test_beam         = args.test_beam
    self.test_word_penalty = args.test_word_penalty
    self.test_gen_limit    = args.test_gen_limit
    # Load in the real data
    log.info("Loading Data")
    self.data_manager.load_train(src=args.src, trg=args.trg,
                                 src_voc=self.nmtrain_model.src_vocab,
                                 trg_voc=self.nmtrain_model.trg_vocab,
                                 src_dev=args.src_dev, trg_dev=args.trg_dev,
                                 src_test=args.src_test, trg_test=args.trg_test,
                                 batch_size=args.batch, unk_cut=args.unk_cut,
                                 src_max_vocab=args.src_max_vocab,
                                 trg_max_vocab=args.trg_max_vocab,
                                 max_sent_length=args.max_sent_length)
    log.info("Loading Finished.")
    # Finalize the model, according to the data
    self.nmtrain_model.finalize_model()

  def train(self, classifier):
    state   = self.nmtrain_model.training_state
    # Watcher is the supervisor of this training
    # It will watch and record everything happens during training
    watcher = nmtrain.watcher.TrainingWatcher(state,
                                              self.nmtrain_model.src_vocab,
                                              self.nmtrain_model.trg_vocab,
                                              self.data_manager.total_trg_words(),
                                              self.early_stop_num)
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
      model.cleargrads()
      batch_loss.backward()
      batch_loss.unchain_backward()
      optimizer.update()

    # Before Training Describe the model
    self.nmtrain_model.describe()

    # Training with maximum likelihood estimation
    data.arrange(state.batch_indexes)
    for ep in range(state.finished_epoch, self.maximum_epoch):
      ep_arrangement = data.shuffle()
      watcher.begin_epoch()
      for src_batch, trg_batch in data.train_data:
        assert(src_batch.id == trg_batch.id)
        # Prepare for training
        batch_loss = classifier.train(model, src_batch, trg_batch,
                                      bptt=bptt,
                                      bptt_len=self.bptt_len)
        watcher.batch_update(loss=batch_loss.data,
                             batch_size=len(trg_batch.data[0]),
                             col_size=len(trg_batch.data)-1)
        bptt(batch_loss)
      watcher.end_epoch(ep_arrangement)

      # Evaluation on Development set
      if data.has_dev_data():
        nmtrain.environment.set_test()
        watcher.begin_evaluation()
        for src_sent, trg_sent in data.dev_data:
          # Prepare for evaluation
          watcher.start_prediction()
          loss = classifier.eval(model, src_sent, trg_sent)
          # TODO(philip30): If we want to do prediction during dev-set 
          # call the prediction method here
          watcher.end_prediction(loss = loss)
        watcher.end_evaluation(data.src_dev, data.trg_dev, self.nmtrain_model.trg_vocab)
        nmtrain.environment.set_train()

      # Incremental testing if wished
      if data.has_test_data():
        nmtrain.environment.set_test()
        tester = nmtrain.Tester(data=data, watcher=test_watcher,
                                trg_vocab=self.nmtrain_model.trg_vocab,
                                classifier=classifier,
                                predict=True, eval_ppl=True)
        tester.test(model = model,
                    word_penalty = self.test_word_penalty,
                    beam_size = self.test_beam,
                    gen_limit = self.test_gen_limit)
        nmtrain.environment.set_train()

      # Check if dev perplexities decline
      dev_ppl_decline = False
      if len(state.dev_perplexities) >= 2:
        dev_ppl_decline = state.dev_perplexities[-1] > state.dev_perplexities[-2]

      # SGD Decay
      if ep + 1 >= self.sgd_lr_decay_after or dev_ppl_decline:
        if optimizer.__class__.__name__ == "SGD":
          optimizer.lr *= self.sgd_lr_decay_factor
          nmtrain.log.info("SGD LR:", optimizer.lr)
        elif optimizer.__class__.__name__ == "Adam":
          optimizer.alpha *= self.sgd_lr_decay_factor
          nmtrain.log.info("Adam Alpha:", optimizer.alpha)

      # Save the model incrementally if wished
      if self.save_models:
        nmtrain.serializer.save(self.nmtrain_model, self.model_file + "-" +
                                str(state.finished_epoch))

      # Stop Early, otherwise, save
      if watcher.should_early_stop():
        break
      elif watcher.should_save() and not self.save_models:
        nmtrain.serializer.save(self.nmtrain_model, self.model_file)

