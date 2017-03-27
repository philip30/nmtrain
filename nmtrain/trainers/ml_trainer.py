import numpy
import gc
import math

import nmtrain
import nmtrain.data
import nmtrain.model
import nmtrain.serializer
import nmtrain.watcher
import nmtrain.reporter
import nmtrain.log as log

class MaximumLikelihoodTrainer:
  def __init__(self, args):
    # If init_model is provided, args will be overwritten
    # Inside the constructor of NmtrainModel using the 
    # previous training specification.
    self.nmtrain_model = nmtrain.NmtrainModel(args)
    self.data_manager  = nmtrain.data.DataManager()
    # Specification
    specification       = self.nmtrain_model.specification
    # Training Parameters
    self.maximum_epoch  = specification.epoch
    self.bptt_len       = specification.bptt_len
    self.early_stop_num = specification.early_stop
    self.save_models    = specification.save_models
    # Unknown Trainers
    self.unknown_trainer = nmtrain.data.unknown_trainer.from_string(specification.unknown_training)
    # Location of output model
    self.model_file = specification.model_out
    # SGD lr decay factor
    self.sgd_lr_decay_factor = specification.sgd_lr_decay_factor
    self.sgd_lr_decay_after  = specification.sgd_lr_decay_after
    # Testing configuration
    self.test_beam         = specification.test_beam
    self.test_word_penalty = specification.test_word_penalty
    self.test_gen_limit    = specification.test_gen_limit
    # Load in the real data
    log.info("Loading Data")
    self.data_manager.load_train(src              = specification.src,
                                 trg              = specification.trg,
                                 src_voc          = self.nmtrain_model.src_vocab,
                                 trg_voc          = self.nmtrain_model.trg_vocab,
                                 src_dev          = specification.src_dev,
                                 trg_dev          = specification.trg_dev,
                                 src_test         = specification.src_test,
                                 trg_test         = specification.trg_test,
                                 batch_size       = specification.batch,
                                 unk_cut          = specification.unk_cut,
                                 src_max_vocab    = specification.src_max_vocab,
                                 trg_max_vocab    = specification.trg_max_vocab,
                                 max_sent_length  = specification.max_sent_length,
                                 sort_method      = specification.sort_method,
                                 batch_strategy   = specification.batch_strategy,
                                 unknown_trainer  = self.unknown_trainer,
                                 bpe_codec        = self.nmtrain_model.bpe_codec)
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
                                              self.early_stop_num)
    # Reporter
    reporter = nmtrain.reporter.TrainingReporter(self.nmtrain_model.specification,
                                                 self.nmtrain_model.src_vocab,
                                                 self.nmtrain_model.trg_vocab)
    # The original chainer model
    model   = self.nmtrain_model.chainer_model
    # Our data manager
    data    = self.data_manager
    # Chainer optimizer
    optimizer = self.nmtrain_model.optimizer
    # Save function
    save = lambda suffix: nmtrain.serializer.save(self.nmtrain_model, self.model_file + suffix)
    # Snapshot save function
    snapshot_counter = 0
    snapshot_threshold = self.nmtrain_model.specification.save_snapshot

    # If test data is provided, prepare the appropriate watcher
    if data.has_test_data:
      self.test_state = nmtrain.model.TestState()
      test_watcher = nmtrain.watcher.TestWatcher(self.test_state,
                                                 self.nmtrain_model.src_vocab,
                                                 self.nmtrain_model.trg_vocab)

    def bptt(batch_loss):
      """ Backpropagation through time """
      if math.isnan(float(batch_loss.data)):
        nmtrain.log.warning("Loss is NaN, skipping update.")
        return

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
      for batch_retriever in self.unknown_trainer:
        for batch in data.train_data:
          src_batch, trg_batch = batch_retriever(batch)
          # Reporting placeholder
          if not reporter.is_reporting:
            output_buffer = None
          else:
            output_buffer = numpy.zeros_like(trg_batch, dtype=numpy.int32)

          # Prepare for training
          watcher.batch_begin()
          batch_loss = classifier.train(model, src_batch, trg_batch,
                                        bptt=bptt,
                                        bptt_len=self.bptt_len,
                                        output_buffer=output_buffer)
          # Generate summary of batch training and keep track of it
          watcher.batch_update(loss=batch_loss.data,
                               batch_size=len(trg_batch[0]),
                               col_size=len(trg_batch)-1,
                               id=batch.id)
          # Report per sentence training if wished
          reporter.train_report(src_batch, trg_batch, output_buffer)
          # BPTT
          bptt(batch_loss)

          # Saving snapshots
          if snapshot_threshold > 0:
            snapshot_counter += len(trg_batch.data[0])
            if snapshot_counter > snapshot_threshold:
              snapeshot_counter = 0
              save("-snapshot")

      watcher.end_epoch(ep_arrangement)

      gc.collect()

      # Evaluation on Development set
      if data.has_dev_data:
        nmtrain.environment.set_test()
        watcher.begin_evaluation()
        for batch in data.dev_data:
          src_sent, trg_sent = batch.normal_data
          # Prepare for evaluation
          watcher.start_prediction()
          loss = classifier.eval(model, src_sent, trg_sent)
          # TODO(philip30): If we want to do prediction during dev-set 
          # call the prediction method here
          watcher.end_prediction(loss = loss)
        watcher.end_evaluation(data.dev_data, self.nmtrain_model.trg_vocab)
        nmtrain.environment.set_train()

      # Incremental testing if wished
      if data.has_test_data:
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
        save("-" + str(state.finished_epoch))

      # Stop Early, otherwise, save
      if watcher.should_early_stop():
        break
      elif watcher.should_save() and not self.save_models:
        save("")

