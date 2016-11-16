import chainer
import numpy

import nmtrain
import nmtrain.data
import nmtrain.chner
import nmtrain.models
import nmtrain.watcher
import nmtrain.log as log

class MaximumLikelihoodTrainer:
  def __init__(self, args):
    self.nmtrain_model = nmtrain.NmtrainModel(args)
    self.data_manager  = nmtrain.data.DataManager()
    # Training Parameters
    self.maximum_epoch = args.epoch
    self.bptt_len = args.bptt_len
    # Load in the real data
    log.info("Loading Data")
    self.data_manager.load_train(args.src, args.trg,
                                 self.nmtrain_model.src_vocab,
                                 self.nmtrain_model.trg_vocab,
                                 args.src_dev, args.trg_dev,
                                 args.batch)
    log.info("Loading Finished.")
    # Finalize the model, according to the data
    self.nmtrain_model.finalize_model(args)

  def train(self):
    xp      = nmtrain.environment.array_module()
    state   = self.nmtrain_model.training_state
    watcher = nmtrain.watcher.TrainingWatcher(state)
    model   = self.nmtrain_model.chainer_model
    data    = self.data_manager
    optimizer = self.nmtrain_model.optimizer

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
        model.reset_state()

        # First encode the sentence
        model.encode(src_data)
        # Predict the target word one step at a time
        # TODO(philip30): Implement Truncated BPTT here.
        batch_loss  = 0
        for trg_word in trg_data:
          y_t = chainer.Variable(trg_word)
          y = model.decode()

          batch_loss += nmtrain.chner.cross_entropy(y, y_t)
          model.update(y_t)
        batch_loss /= len(trg_batch.data)
        watcher.batch_update(loss=batch_loss.data, size=len(trg_batch.data[0]))
        # BPTT
        model.zerograds()
        batch_loss.backward()
        optimizer.update()
      watcher.end_epoch()

      # TODO(philip30): Implement Evaluation here
    return
