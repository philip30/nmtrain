import atexit
import chainer
import os
import pickle
import shutil
import tempfile
import zipfile

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
  if not out_file.endswith(".zip"):
    out_file = out_file + ".zip"

  tmpdir = tempfile.mkdtemp()
  log.info("Saving model to", out_file)

  # Saving Specification
  pickle_save(os.path.join(tmpdir, SPEC), model.specification.__dict__)

  # Saving vocabularies
  pickle_save(os.path.join(tmpdir, SRC_VOC), model.src_vocab)
  pickle_save(os.path.join(tmpdir, TRG_VOC), model.trg_vocab)

  # Saving optimizer state
  chainer.serializers.save_npz(os.path.join(tmpdir, OPTIMIZER), model.optimizer)

  # Saving training state
  pickle_save(os.path.join(tmpdir, STATE), model.training_state)

  # Saving Weight
  chainer.serializers.save_npz(os.path.join(tmpdir, WEIGHT), model.chainer_model)

  # Zipping
  zf = zipfile.ZipFile(out_file, mode="w", compression=zipfile.ZIP_DEFLATED)
  try:
    write_zip(zf, os.path.join(tmpdir, SRC_VOC))
    write_zip(zf, os.path.join(tmpdir, TRG_VOC))
    write_zip(zf, os.path.join(tmpdir, STATE))
    write_zip(zf, os.path.join(tmpdir, WEIGHT))
    write_zip(zf, os.path.join(tmpdir, SPEC))
    write_zip(zf, os.path.join(tmpdir, OPTIMIZER))
  finally:
    zf.close()

  log.info("Finished saving model.")
  atexit.register(lambda dir=tmpdir: shutil.rmtree(tmpdir))

def load(model, in_file):
  tmpdir = tempfile.mkdtemp()
  zf = zipfile.ZipFile(in_file, mode="r")
  for filename in zf.namelist():
    zf.extract(filename, tmpdir)
  model.specification = lambda: None
  model.specification.__dict__.update(pickle_load(os.path.join(tmpdir, SPEC)))
  # Update Seed
  nmtrain.environment.init_random(model.specification.seed)

  # Loading vocabularies
  model.src_vocab = pickle_load(os.path.join(tmpdir, SRC_VOC))
  model.trg_vocab = pickle_load(os.path.join(tmpdir, TRG_VOC))

  # Loading training state
  model.training_state = pickle_load(os.path.join(tmpdir, STATE))

  # Loading Weight
  model.chainer_model = nmtrain.model.from_spec(model.specification,
                                                len(model.src_vocab),
                                                len(model.trg_vocab))
  if nmtrain.environment.is_train():
    model.optimizer = nmtrain.model.parse_optimizer(model.specification.optimizer)
    model.optimizer.setup(model.chainer_model)
  chainer.serializers.load_npz(os.path.join(tmpdir, WEIGHT), model.chainer_model)

  # Loading Optimizer
  if nmtrain.environment.is_train():
    chainer.serializers.load_npz(os.path.join(tmpdir, OPTIMIZER), model.optimizer)
  atexit.register(lambda dir=tmpdir: shutil.rmtree(tmpdir))

#####################
# PRIVATE FUNCTIONS #
#####################
def pickle_save(file_out, obj):
  with open(file_out, "wb") as fp:
    pickle.dump(obj, fp)

def pickle_load(file_in):
  with open(file_in, "rb") as fp:
    return pickle.load(fp)

def write_zip(zipobj, path):
  zipobj.write(path, os.path.basename(path))

