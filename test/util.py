import tempfile
import os

import nmtrain
import nmtrain.log as log

def init_env(gpu=-1, seed=2):
  nmtrain.environment.init_gpu(-1)
  nmtrain.environment.init_random(3)

def basic_train_args(model_architecture="encdec"):
  args = lambda: None
  args.model_architecture = "encdec"
  args.embed  = 50
  args.hidden = 50
  args.dropout = 0.1
  args.depth = 1
  args.epoch = 1
  args.bptt_len = 1
  args.batch = 1
  args.optimizer = "adam"
  args.init_model = ""
  args.src_dev = ""
  args.trg_dev = ""
  args.src_test = ""
  args.trg_test = ""
  args.early_stop = 0
  args.unk_cut = 0
  args.max_vocab = 10000000000
  # Dummy test file
  this_script_dir = os.path.dirname(os.path.realpath(__file__))
  args.src = os.path.join(this_script_dir, "data", "train.ja")
  args.trg = os.path.join(this_script_dir, "data", "train.en")
  args.model_out = temp_model_file()
  return args

def basic_test_args(init_model):
  args = lambda: None
  args.init_model = init_model
  return args

def temp_model_file(name="nmtrain.model"):
  model_file = tempfile.gettempdir() + "/" + name
  log.info("Model file is requested at:", model_file)
  return model_file
