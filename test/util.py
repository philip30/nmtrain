import tempfile
import os

import nmtrain
import nmtrain.log as log

def init_env(gpu=-1, seed=2):
  nmtrain.environment.init_gpu(-1)
  nmtrain.environment.init_random(3)

def basic_train_args(model_architecture="encdec", src="train.ja", trg="train.en"):
  class Args(object):
    pass
  args = Args()
  args.model_architecture = "encdec"
  args.embed  = 50
  args.hidden = 50
  args.dropout = 0.1
  args.depth = 1
  args.epoch = 1
  args.bptt_len = 1
  args.batch = 1
  args.optimizer = "adam"
  args.init_model = None
  args.src_dev = None
  args.trg_dev = None
  args.src_test = None
  args.trg_test = None
  args.early_stop = 0
  args.unk_cut = 0
  args.max_vocab = 10000000000
  args.src_bpe_codec = ""
  args.trg_bpe_codec = ""
  args.save_models = False
  args.unknown_training = "normal"
  args.sgd_lr_decay_factor = 1.0
  args.sgd_lr_decay_after = 10
  args.test_beam = 1
  args.test_word_penalty = 0.1
  args.test_gen_limit = 20
  args.src_max_vocab = 1000
  args.trg_max_vocab = 1000
  args.max_sent_length = 20
  args.sort_method = "lentrg"
  args.batch_strategy = "sentence"
  args.gradient_clipping = 5.0
  args.save_snapshot = 10000
  args.seed = -1
  args.lexicon = None

  # Dummy test file
  this_script_dir = os.path.dirname(os.path.realpath(__file__))
  args.src = os.path.join(this_script_dir, "data", src)
  args.trg = os.path.join(this_script_dir, "data", trg)
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
