#!/usr/bin/env python3

import argparse

import nmtrain.log as log
import nmtrain.enumeration
import nmtrain.classifiers
import nmtrain.trainers

""" Arguments """
parser = argparse.ArgumentParser("NMT model trainer")
# Required
parser.add_argument("--init_model", type=str, required=True)
parser.add_argument("--orig_src", type=str, required=True)
parser.add_argument("--orig_trg", type=str, required=True)
parser.add_argument("--dest_src", type=str, default=None)
parser.add_argument("--dest_trg", type=str, default=None)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

def main(args):
  # Initiation
  sanity_check(args)
  nmtrain.environment.init(args, nmtrain.enumeration.RunMode.TRAIN)

  # Load up data
  trainer = nmtrain.trainers.SequenceGANTrainer(args)
  trainer.train(nmtrain.classifiers.RNN_NMT())

def sanity_check(args):
  pass

if __name__ == "__main__":
  main(args)
