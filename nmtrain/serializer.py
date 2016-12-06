import os
import chainer
import pickle

import nmtrain.model
import nmtrain.log as log

# File names
SPEC      = "mod.spec"
OPTIMIZER = "mod.optimizer"
SRC_VOC   = "src.vocab"
TRG_VOC   = "trg.vocab"
STATE     = "mod.state"
WEIGHT    = "mod.weight"

def save(model, out_file):
  log.info("Saving model...")
  # Init directory
  if not os.path.exists(out_file):
    os.makedirs(out_file)

  if model.training_state.finished_epoch == 1:
    # Saving Specification
    pickle_save(os.path.join(out_file, SPEC), model.specification.__dict__)

    # Saving vocabularies
    pickle_save(os.path.join(out_file, SRC_VOC), model.src_vocab)
    pickle_save(os.path.join(out_file, TRG_VOC), model.trg_vocab)

  # Saving optimizer state
  chainer.serializers.save_npz(os.path.join(out_file, OPTIMIZER), model.optimizer)

  # Saving training state
  pickle_save(os.path.join(out_file, STATE), model.training_state)

  # Saving Weight
  chainer.serializers.save_npz(os.path.join(out_file, WEIGHT), model.chainer_model)
  log.info("Finished saving model.")

def load(model, in_file):
  # Loading Specification
  model.specification = lambda: None
  model.specification.__dict__.update(pickle_load(os.path.join(in_file, SPEC)))
  # Update Seed
  nmtrain.environment.init_random(model.specification.seed)

  # Loading vocabularies
  model.src_vocab = pickle_load(os.path.join(in_file, SRC_VOC))
  model.trg_vocab = pickle_load(os.path.join(in_file, TRG_VOC))

  # Loading training state
  model.training_state = pickle_load(os.path.join(in_file, STATE))

  # Loading Weight
  model.chainer_model = nmtrain.model.from_spec(model.specification,
                                                len(model.src_vocab),
                                                len(model.trg_vocab))
  if nmtrain.environment.is_train():
    model.optimizer = nmtrain.model.parse_optimizer(model.specification.optimizer)
    model.optimizer.setup(model.chainer_model)
  chainer.serializers.load_npz(os.path.join(in_file, WEIGHT), model.chainer_model)

  # Loading Optimizer
  if nmtrain.environment.is_train():
    chainer.serializers.load_npz(os.path.join(in_file, OPTIMIZER), model.optimizer)

#####################
# PRIVATE FUNCTIONS #
#####################
def pickle_save(file_out, obj):
  with open(file_out, "wb") as fp:
    pickle.dump(obj, fp)

def pickle_load(file_in):
  with open(file_in, "rb") as fp:
    return pickle.load(fp)

