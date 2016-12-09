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
parser.add_argument("--init_model", type=str, required=True, help="Init the model with the pretrained model.")
parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level.")
parser.add_argument("--gen_limit", type=int, default=50, help="Maximum Target Output Length.")
parser.add_argument("--beam", type=int, default=1, help="Beam size in searching.")
parser.add_argument("--word_penalty", type=float, default=0.0, help="Word penalty in beam search")
# Evaluation
parser.add_argument("--ref", type=str, default=None)
args = parser.parse_args()

def main(args):
  sanity_check(args)
  nmtrain.environment.init(args, nmtrain.enumeration.RunMode.TEST)

  # Manager of batches and data
  data_manager  = nmtrain.data.DataManager()
  # The model, chainer model inside
  model         = nmtrain.NmtrainModel(args)
  model.finalize_model()
  model.describe()
  # The watcher, who logs everything
  watcher       = nmtrain.TestWatcher(state         = nmtrain.model.TestState(),
                                      src_vocab     = model.src_vocab,
                                      trg_vocab     = model.trg_vocab,
                                      output_stream = sys.stdout )
  # PostProcessor
  post_processor = nmtrain.NMTrainPostProcessor()
  # Classifier, that run the data and the model
  classifier    = nmtrain.classifiers.RNN_NMT()

   # Array module
  xp = nmtrain.environment.array_module()
  log.info("Loading Data")
  data_manager.load_test(src     = args.src,
                         src_voc = model.src_vocab,
                         trg_voc = model.trg_vocab,
                         ref     = args.ref)
  log.info("Loading Finished.")

  # Begin Testing
  tester = nmtrain.Tester(data=data_manager, watcher=watcher,
                          trg_vocab=model.trg_vocab,
                          classifier=classifier,
                          predict=True, eval_ppl=(args.ref is not None))
  tester.test(model = model.chainer_model,
              word_penalty = args.word_penalty,
              beam_size = args.beam,
              gen_limit = args.gen_limit)

def sanity_check(args):
  pass

if __name__ == "__main__":
  main(args)
