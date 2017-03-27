#!/usr/bin/env python3

import argparse
import numpy
import sys

import nmtrain
import nmtrain.model
import nmtrain.classifiers
import nmtrain.log as log

""" Arguments """
parser = argparse.ArgumentParser("NMT decoder")
# Required
parser.add_argument("--src", type=str, required=True)
# Configuration
parser.add_argument("--gpu", type=int, default=-1, help="Specify GPU to be used, negative for using CPU.")
parser.add_argument("--init_model", type=str, nargs="+", required=True, help="Init the model with the pretrained model.")
parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level.")
parser.add_argument("--gen_limit", type=int, default=50, help="Maximum Target Output Length.")
parser.add_argument("--beam", type=int, default=1, help="Beam size in searching.")
parser.add_argument("--word_penalty", type=float, default=0.0, help="Word penalty in beam search")
parser.add_argument("--memory_optimization", type=int, default=0)
# Ensemble
parser.add_argument("--ensemble_op", type=str, choices=["linear", "logsum"], default="linear")
# Evaluation
parser.add_argument("--ref", type=str, default=None)
args = parser.parse_args()

def main(args):
  sanity_check(args)
  nmtrain.environment.init(args, nmtrain.enumeration.RunMode.TEST)

  # Manager of batches and data
  data_manager  = nmtrain.data.DataManager()
  # The model, chainer model inside
  model         = load_model(args)
  # The watcher, who logs everything
  watcher       = nmtrain.TestWatcher(state         = nmtrain.model.TestState(),
                                      src_vocab     = model.src_vocab,
                                      trg_vocab     = model.trg_vocab,
                                      output_stream = sys.stdout )
  # Classifier, that run the data and the model
  classifier    = nmtrain.classifiers.RNN_NMT()

  print("~~ Testing ~~", file=sys.stderr)
  print("gen_limit:", args.gen_limit, file=sys.stderr)
  print("beam:", args.beam, file=sys.stderr)
  print("word_penalty:", args.word_penalty, file=sys.stderr)
  print("gpu:", args.gpu, file=sys.stderr)

  log.info("Loading Data")
  data_manager.load_test(src     = args.src,
                         src_voc = model.src_vocab,
                         trg_voc = model.trg_vocab,
                         ref     = args.ref,
                         bpe_codec = model.bpe_codec)
  log.info("Loading Finished.")

  # Begin Testing
  tester = nmtrain.Tester(data=data_manager, watcher=watcher,
                          trg_vocab=model.trg_vocab,
                          classifier=classifier,
                          predict=True, eval_ppl=(args.ref is not None))

  if model.__class__.__name__ == "NmtrainModel":
    model = model.chainer_model

  tester.test(model = model,
              word_penalty = args.word_penalty,
              beam_size = args.beam,
              gen_limit = args.gen_limit)

def sanity_check(args):
  pass

def load_model(args):
  models = args.init_model

  def load_single_model(spec):
    model         = nmtrain.NmtrainModel(spec)
    model.finalize_model()
    model.describe()
    return model

  if len(models) == 1:
    args.init_model = args.init_model[0]
    ret = load_single_model(args)
  else:
    ret = []
    for model in models:
      args.init_model = model
      ret.append(load_single_model(args))
    if args.ensemble_op == "linear":
      ret = nmtrain.models.EnsembleLinearInterpolateNMT(ret)
    elif args.ensemble_op == "logsum":
      ret = nmtrain.models.EnsembleLogSumNMT(ret)
  args.init_model = models
  return ret

if __name__ == "__main__":
  main(args)
