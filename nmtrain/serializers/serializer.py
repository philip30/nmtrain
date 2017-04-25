import atexit
import chainer
import os
import pickle
import shutil
import tempfile
import zipfile
import numpy

import nmtrain
import nmtrain.log as log

# File names
SPEC      = "mod.spec"
OPTIMIZER = "mod.optimizer"
SRC_VOC   = "src.vocab"
TRG_VOC   = "trg.vocab"
STATE     = "mod.state"
WEIGHT    = "mod.weight"
LEXICON   = "mod.lexicon"
BPE_CODEC = "mod.bpe_codec"
NUMPY_RANDOM = "numpy.state"
CUDA_RANDOM = "cuda.state"

def save(model, out_file):
  tmpdir = tempfile.mkdtemp()
  log.info("Saving model to", out_file)

  # Saving Specification
  proto_save(os.path.join(tmpdir, SPEC), model.config)

  # Saving vocabularies
  proto_save(os.path.join(tmpdir, SRC_VOC), model.src_vocab.data)
  proto_save(os.path.join(tmpdir, TRG_VOC), model.trg_vocab.data)

  # Saving optimizer state
  chainer.serializers.save_npz(os.path.join(tmpdir, OPTIMIZER), model.optimizer)

  # Saving training state
  proto_save(os.path.join(tmpdir, STATE), model.state.data)
  pickle_save(os.path.join(tmpdir, NUMPY_RANDOM), numpy.random.get_state())
  if hasattr(chainer.cuda, "cupy"):
    pickle_save(os.path.join(tmpdir, CUDA_RANDOM), chainer.cuda.cupy.random.get_state())

  # Saving Weight
  chainer.serializers.save_npz(os.path.join(tmpdir, WEIGHT), model.chainer_model)

  # Saving Lexicon
  if model.lexicon is not None:
    proto_save(os.path.join(tmpdir, LEXICON), model.lexicon.data)

  # Saving BPE codec
  if hasattr(model, "bpe_codec"):
    pickle_save(os.path.join(tmpdir, BPE_CODEC), model.bpe_codec)

  # Zipping
  zf = zipfile.ZipFile(out_file, mode="w", compression=zipfile.ZIP_DEFLATED)
  try:
    write_zip(zf, os.path.join(tmpdir, SRC_VOC))
    write_zip(zf, os.path.join(tmpdir, TRG_VOC))
    write_zip(zf, os.path.join(tmpdir, STATE))
    write_zip(zf, os.path.join(tmpdir, WEIGHT))
    write_zip(zf, os.path.join(tmpdir, SPEC))
    write_zip(zf, os.path.join(tmpdir, OPTIMIZER))
    write_zip(zf, os.path.join(tmpdir, NUMPY_RANDOM))
    if model.lexicon is not None: write_zip(zf, os.path.join(tmpdir, LEXICON))
    if hasattr(model, "bpe_codec"): write_zip(zf, os.path.join(tmpdir, BPE_CODEC))
    if hasattr(chainer.cuda, "cupy"): write_zip(zf, os.path.join(tmpdir, CUDA_RANDOM))
  finally:
    zf.close()

  log.info("Finished saving model.")
  atexit.register(lambda dir=tmpdir: shutil.rmtree(tmpdir))

def load(reader, from_config):
  tmpdir = tempfile.mkdtemp()
  zf = zipfile.ZipFile(from_config.init_model, mode="r")
  for filename in zf.namelist():
    zf.extract(filename, tmpdir)
  atexit.register(lambda dir=tmpdir: shutil.rmtree(tmpdir))

  # First extract the proto
  config = proto_load(os.path.join(tmpdir, SPEC), nmtrain.train_config_pb.TrainingConfig())
  reader.load_config(config, from_config)

  # Loading vocabularies
  src_vocab = proto_load(os.path.join(tmpdir, SRC_VOC), nmtrain.dictionary_pb.Vocabulary())
  trg_vocab = proto_load(os.path.join(tmpdir, TRG_VOC), nmtrain.dictionary_pb.Vocabulary())
  reader.load_vocabularies(src_vocab, trg_vocab)

  # Loading training state
  state = proto_load(os.path.join(tmpdir, STATE), nmtrain.state_pb.NmtrainState())
  numpy_state = os.path.join(tmpdir, NUMPY_RANDOM)
  cuda_state = os.path.join(tmpdir, CUDA_RANDOM)
  reader.load_state(state)
  numpy.random.set_state(pickle_load(numpy_state))
  if hasattr(chainer.cuda, "cupy"):
    chainer.cuda.cupy.set_state(pickle_load(cuda_state))

  # Loading Lexicon
  lexicon_path = os.path.join(tmpdir, LEXICON)
  if os.path.exists(lexicon_path):
    reader.load_lexicon(proto_load(lexicon_path, nmtrain.dictionary_pb.Lexicon()))
  else:
    reader.load_lexicon(None)

  # Loading BPE Codec
  bpe_path = os.path.join(tmpdir, BPE_CODEC)
  if os.path.exists(bpe_path):
    reader.load_bpe(pickle_load(bpe_path))

  # Loading Chainer objects
  reader.load_chainer_objects(os.path.join(tmpdir, WEIGHT), os.path.join(tmpdir, OPTIMIZER)) 

#####################
# PRIVATE FUNCTIONS #
#####################
def pickle_save(file_out, obj):
  with open(file_out, "wb") as fp:
    pickle.dump(obj, fp)

def pickle_load(file_in):
  with open(file_in, "rb") as fp:
    return pickle.load(fp)

def proto_save(file_out, obj):
  with open(file_out, "wb") as fp:
    fp.write(obj.SerializeToString())

def proto_load(file_in, proto_obj):
  with open(file_in, "rb") as fp:
    proto_obj.ParseFromString(fp.read())
  return proto_obj

def write_zip(zipobj, path):
  zipobj.write(path, os.path.basename(path))

